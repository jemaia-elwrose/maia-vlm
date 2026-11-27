"""The pipeline: event + frames -> images -> prompt -> model -> guarded verdict."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import guard
from .backends import Backend, BackendError
from .checks import Check
from .frames import Event, Frame, Shot, box_to, read_clip, select, shots

Status = Literal["ok", "repaired", "inconsistent", "ungrounded", "no_verdict"]

SYSTEM = (
    "You check alerts from a video analytics system. Judge only what is visible in the images. "
    "If the images do not show enough to decide, answer 'unclear'. Reply with JSON only."
)

BOX_WORDS = {
    "norm1000": "[x1, y1, x2, y2] on a 0-1000 scale of the image's width and height",
    "norm1": "[x1, y1, x2, y2] as fractions (0-1) of the image's width and height",
    "pixels": "[x1, y1, x2, y2] in pixels of the image",
}


class Verdict(BaseModel):
    check: str
    check_version: int
    check_sha256: str
    model: str
    verdict: Literal["confirmed", "rejected", "unclear", "no_verdict"]
    description: str = ""
    answer: dict[str, Any] = Field(default_factory=dict)
    status: Status = "ok"
    reason: str = ""  # why a guardrail fired
    needs_review: bool = False  # review-only checks: a confirmation is a prompt to look, not a finding
    images: list[dict] = Field(default_factory=list)  # what the model was shown
    latency_ms: float = 0.0
    calls: int = 0
    prompt_tokens: int | None = None  # as the server counted them, images included (first call)
    raw: list[str] = Field(default_factory=list)


def subject_text(event: Event, check: Check, images: Sequence[Shot], backend: Backend) -> str:
    label = event.label
    if event.box is None:
        return f"the scene (the detector reported a {label}, without saying where)"
    at = next(n for n, s in enumerate(images, start=1) if s.at_event)
    if check.frames.crop == "subject":
        if len(images) == 1:
            return f"the {label} in the middle of the close-up"
        return (
            f"the {label} in the middle of close-up {at} (taken at the event time); in the others they may have moved"
        )
    shot = images[at - 1]
    box = box_to(backend.box_format, shot.to_image(event.box), shot.size)
    where = f"the {label} at box {box} ({BOX_WORDS[backend.box_format]}) in image {at}, taken at the event time"
    if check.frames.crop == "both":
        return f"{where}; image 1 is the whole scene, the others are close-ups of that {label}"
    return where if len(images) == 1 else f"{where}; in the other images they may have moved"


def evidence_image(images: Sequence[Shot]) -> int:
    """The image evidence is asked for: the first taken at the event time (with crop 'both',
    the whole frame, whose coordinates match the box given in the prompt)."""
    return next(n for n, s in enumerate(images, start=1) if s.at_event)


def zone_text(event: Event, images: Sequence[Shot], backend: Backend) -> str:
    if not event.zone:
        return "no zone given"
    full = next((s for s in images if s.kind == "full"), None)
    if full is None:
        return "a zone (not visible in close-ups)"
    w, h = full.size
    pts = []
    for x, y in event.zone:
        ix, iy = (x - full.x0) * full.scale, (y - full.y0) * full.scale
        b = box_to(backend.box_format, (ix, iy, ix, iy), (w, h))
        pts.append([b[0], b[1]])
    return f"the polygon {pts} (same scale as boxes)"


def build_prompt(check: Check, event: Event, images: Sequence[Shot], backend: Backend) -> str:
    span = images[-1].t_s - images[0].t_s if len(images) > 1 else 0.0
    text = check.render(
        n_frames=len(images),
        time=event.clock or f"{event.at_s or 0:.1f} s into the clip",
        camera=event.camera,
        label=event.label,
        subject=subject_text(event, check, images, backend),
        zone=zone_text(event, images, backend),
        span_s=f"{span:.1f}",
    )
    extra = [f"Images are in time order, {span:.1f} s apart from first to last."] if len(images) > 1 else []
    if "evidence" in check.answer.get("properties", {}):
        k = evidence_image(images)
        extra.append(
            f"evidence: at most one item, on image {k} (taken at the event time): 'image' is {k} and "
            f"'box' surrounds what decided your answer, as {BOX_WORDS[backend.box_format]}. "
            "Leave evidence empty if nothing in particular decided it."
        )
    extra.append(f"Reply with only a JSON object matching this schema: {guard.schema_text(check.answer)}")
    return text.rstrip() + "\n\n" + "\n".join(extra)


def verify(
    event: Event, frames: Sequence[Frame], check: Check, backend: Backend, save_to: str | Path | None = None
) -> Verdict:
    """Ask ``check`` about ``event``, given the frames around it. ``save_to``: a folder to write
    the images the model saw, with the detection and its evidence drawn on them (debugging)."""
    base = {"check": check.id, "check_version": check.version, "check_sha256": check.sha256, "model": backend.name}
    if not frames:
        return Verdict(**base, verdict="no_verdict", status="no_verdict", reason="no frames")
    at = event.at_s if event.at_s is not None else frames[len(frames) // 2].t_s
    chosen = select(frames, at, check.frames)
    images = shots(chosen, event.model_copy(update={"at_s": at}), check.frames)
    shown = [{"t_s": round(s.t_s, 3), "kind": s.kind, "size": list(s.size)} for s in images]
    prompt = build_prompt(check, event, images, backend)
    messages = backend.messages(prompt, images, SYSTEM)
    raw: list[str] = []
    latency = 0.0
    tokens: int | None = None
    status: Status = "ok"
    answer: dict | None = None
    problem = ""
    for attempt in range(2):  # the answer, then one repair
        try:
            reply = backend.chat(messages, check.answer, check.temperature, check.max_tokens)
        except BackendError as exc:
            verdict = Verdict(**base, verdict="no_verdict", status="no_verdict", reason=str(exc), images=shown,
                              latency_ms=round(latency + exc.elapsed_ms, 1), calls=attempt, raw=raw)  # fmt: skip
            _save(save_to, images, event, verdict, backend)
            return verdict
        raw.append(reply.text)
        latency += reply.latency_ms
        if tokens is None:
            tokens = reply.usage.get("prompt_tokens")
        try:
            candidate = guard.parse(reply.text)
            errors = guard.validate(candidate, check.answer)
        except guard.ParseError as exc:
            candidate, errors = None, [str(exc)]
        if not errors:
            answer = candidate
            status = "repaired" if attempt else "ok"
            break
        problem = guard.describe_errors(errors)
        messages = [
            *messages,
            {"role": "assistant", "content": reply.text},
            {"role": "user", "content": f"That reply is not valid: {problem}. Reply again with only the JSON object."},
        ]
    common = {**base, "images": shown, "latency_ms": round(latency, 1), "calls": len(raw), "raw": raw,
              "prompt_tokens": tokens}  # fmt: skip
    if answer is None:
        verdict = Verdict(**common, verdict="no_verdict", status="no_verdict", reason=f"invalid answer: {problem}")
        _save(save_to, images, event, verdict, backend)
        return verdict
    verdict = answer["verdict"]
    reason = ""
    if check.verdict_rules:
        derived, why = _derive(check, answer)
        if derived != verdict:
            # trust the specific answers, but a contradiction never becomes a confirmation
            final = "unclear" if derived == "confirmed" else derived
            reason = f"verdict '{verdict}' contradicts {why}; using '{final}'"
            status, verdict = "inconsistent", final
    ungrounded = guard.grounding(answer, images, event.box, backend.box_format, check.grounding.min_inside)
    if ungrounded:
        status, verdict, reason = "ungrounded", "unclear", ungrounded
    out = Verdict(
        **common,
        verdict=verdict,
        description=str(answer.get("description", "")),
        answer=answer,
        status=status,
        reason=reason,
        needs_review=check.review_only and verdict == "confirmed",
    )
    _save(save_to, images, event, out, backend)
    return out


def _derive(check: Check, answer: dict) -> tuple[str, str]:
    """Confirmed only if every rule says confirmed; otherwise rejected beats unclear."""
    results = [(rule.map[answer[rule.field]], f"{rule.field}='{answer[rule.field]}'") for rule in check.verdict_rules]
    for wanted in ("rejected", "unclear"):
        hits = [why for got, why in results if got == wanted]
        if hits:
            return wanted, ", ".join(hits)
    return "confirmed", ", ".join(why for _, why in results)


def _save(folder: str | Path | None, images: Sequence[Shot], event: Event, verdict: Verdict, backend: Backend) -> None:
    if folder is not None:
        from .debug import save_shots

        save_shots(folder, images, event, verdict, backend.box_format)


def thin(frames: Sequence[Frame], fps: float) -> list[Frame]:
    """Keep frames ``1/fps`` apart, as maia-gateway's sampling does before frames reach a ring."""
    out: list[Frame] = []
    for f in frames:
        if not out or f.t_s - out[-1].t_s >= 1 / fps - 1e-3:
            out.append(f)
    return out


def verify_clip(
    path: str | Path,
    event: Event,
    check: Check,
    backend: Backend,
    sample_fps: float | None = None,
    save_to: str | Path | None = None,
) -> Verdict:
    """Read the frames a check needs from a clip, then verify. ``sample_fps`` thins them first,
    to see what a live check gets from a ring filled at that rate (maia-gateway's default: 5)."""
    from .frames import clip_duration

    at = event.at_s if event.at_s is not None else clip_duration(path) / 2
    lo, hi = check.frames.window_s
    frames = read_clip(path, max(0.0, at + lo), at + hi)
    if sample_fps:
        frames = thin(frames, sample_fps)
    return verify(event.model_copy(update={"at_s": at}), frames, check, backend, save_to)
