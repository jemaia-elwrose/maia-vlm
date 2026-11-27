"""Run a list of cases (clip + check + event) and keep every verdict, for testing and timing.

cases.yaml:

    media: media                 # folder of the clips, relative to this file
    defaults: {fps: 5, at: 5}    # applied to every case unless it says otherwise
    cases:
      - {id: ppe-helmet, clip: ppe.mp4, check: helmet, box: [581, 371, 1113, 901], expect: rejected}
      - {id: shop-describe, clip: shop.mp4, check: describe}

``expect`` is optional: cases with it count towards accuracy, the others are just timed.
``box: auto`` asks a maia-infer YOLO worker (``--infer``) for the largest detection of the
case's label at its time (``auto-group``: one box around all of them), so a set of public clips
needs no boxes drawn by hand; the box used is saved with each result. Whole-frame checks run
without a box when none is found. ``at`` defaults to the middle of the clip.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import checks as checks_mod
from .backends import Backend
from .frames import Event
from .verify import Verdict, verify_clip


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    clip: str
    check: str
    label: str = "person"
    box: tuple[float, float, float, float] | Literal["auto", "auto-group"] | None = None
    at: float | None = None
    fps: float | None = None
    zone: list[tuple[float, float]] | None = None
    expect: Literal["confirmed", "rejected", "unclear"] | None = None
    note: str = ""


class CaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    media: str = "media"
    defaults: dict[str, Any] = Field(default_factory=dict)
    cases: list[dict[str, Any]]


def load_cases(path: str | Path, media: str | Path | None = None) -> tuple[list[Case], Path]:
    path = Path(path)
    spec = CaseFile.model_validate(yaml.safe_load(path.read_text()))
    cases = [Case.model_validate({**spec.defaults, **c}) for c in spec.cases]
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate case ids: {sorted(dupes)}")
    folder = Path(media) if media else (path.parent / spec.media)
    return cases, folder


class Detector:
    """Boxes for ``box: auto`` / ``auto-group``, from a maia-infer detection worker (RF-DETR by
    default; BGR input, COCO names). ``auto``: the largest detection of the label, the main subject of a test clip (the
    most confident one is often a bystander, or a false positive on furniture). ``auto-group``:
    one box around all of them, for checks about a group (a fight). When the case's frame has
    none, the frames up to 0.8 s around it are tried, nearest first."""

    OFFSETS = (0.0, -0.4, 0.4, -0.8, 0.8)

    def __init__(self, endpoint: str, service: str = "rfdetr", conf: float = 0.3):
        from maia_infer import Client  # optional: only sets with box: auto need it

        self.client, self.service, self.conf = Client(endpoint, timeout_ms=20000), service, conf

    def box(self, clip: Path, at: float, label: str, mode: str = "auto") -> tuple[float, float, float, float] | None:
        from maia_infer import detections

        from .frames import read_clip

        frames = read_clip(clip, max(0.0, at - 1.0), at + 1.0)
        for off in self.OFFSETS:
            if not frames:
                return None
            frame = min(frames, key=lambda f: abs(f.t_s - (at + off)))
            if abs(frame.t_s - (at + off)) > 0.25:
                continue
            bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
            found = [d["box"] for d in detections(self.client.infer(self.service, bgr, meta={"conf": self.conf}))
                     if d["name"] == label and d["conf"] >= self.conf]  # fmt: skip
            if not found:
                continue
            if mode == "auto-group":
                box = (
                    min(b[0] for b in found),
                    min(b[1] for b in found),
                    max(b[2] for b in found),
                    max(b[3] for b in found),
                )
            else:
                box = max(found, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            return tuple(round(v, 1) for v in box)  # type: ignore[return-value]
        return None

    def close(self) -> None:
        self.client.close()


def run_case(
    case: Case,
    folder: Path,
    backend: Backend,
    check_dirs: tuple = (),
    save_to: Path | None = None,
    detector: Detector | None = None,
) -> tuple[Verdict, dict]:
    """The verdict, and what the case resolved to (the time and box used)."""
    from .frames import clip_duration

    check = checks_mod.find(case.check, *check_dirs)
    clip = folder / case.clip
    at = case.at if case.at is not None else round(clip_duration(clip) / 2, 2)
    box = case.box
    used = {"at": at, "box_from": "given" if isinstance(box, tuple) else (box or "none")}
    if box in ("auto", "auto-group"):
        if detector is None:
            raise ValueError(f"case {case.id} has box: {box}: run with --infer <maia-infer endpoint>")
        box = detector.box(clip, at, case.label, mode=str(box))
        if box is None and check.frames.crop != "full":
            base = {"check": check.id, "check_version": check.version, "check_sha256": check.sha256,
                    "model": backend.name}  # fmt: skip
            reason = f"box: auto found no {case.label} within 0.8 s of {at:.2f} s, and this check crops around it"
            return Verdict(**base, verdict="no_verdict", status="no_verdict", reason=reason), {**used, "box": None}
        if box is None:  # a whole-frame check (fall, fight, describe) can still look at the scene
            used["box_from"] += ": none found, whole scene"
    used["box"] = list(box) if box else None
    event = Event(label=case.label, box=box, at_s=at, zone=case.zone)
    return verify_clip(clip, event, check, backend, sample_fps=case.fps, save_to=save_to), used


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def mark(row: dict) -> str:
    """ok: as expected. no alert: an expected rejection answered unclear or no_verdict, which
    raises no alert either (safe, but undecided). WRONG: a false alarm, or a missed one."""
    exp, got = row.get("expect"), row["verdict"]
    if not exp:
        return ""
    if got == exp:
        return "ok"
    if exp == "rejected" and got in ("unclear", "no_verdict"):
        return "no alert"
    return "WRONG"


def summarize(rows: list[dict]) -> dict:
    """Exact accuracy, and the alert view an operator cares about: an alert is a 'confirmed'."""
    judged = [r for r in rows if r.get("expect")]
    right = [r for r in judged if r["verdict"] == r["expect"]]
    real = [r for r in judged if r["expect"] == "confirmed"]
    fake = [r for r in judged if r["expect"] == "rejected"]
    caught = sum(r["verdict"] == "confirmed" for r in real)
    false_alarms = sum(r["verdict"] == "confirmed" for r in fake)
    lat = [r["latency_ms"] for r in rows if r["calls"]]
    toks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "cases": len(rows),
        "with_expected": len(judged),
        "correct": len(right),
        "accuracy": round(len(right) / len(judged), 3) if judged else None,
        "alerts": {
            "real_events": len(real),
            "caught": caught,
            "missed": len(real) - caught,
            "false_events": len(fake),
            "false_alarms": false_alarms,
            "recall": round(caught / len(real), 3) if real else None,
            "false_alarm_rate": round(false_alarms / len(fake), 3) if fake else None,
        },
        "statuses": statuses,
        "latency_ms_p50": round(_pct(lat, 0.5), 1),
        "latency_ms_p95": round(_pct(lat, 0.95), 1),
        "prompt_tokens_mean": round(statistics.mean(toks)) if toks else None,
    }


def table(rows: list[dict]) -> str:
    head = f"{'case':<26} {'check':<18} {'expect':<10} {'verdict':<11} {'status':<12} {'ms':>7} {'tokens':>6}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['case'][:26]:<26} {r['check'][:18]:<18} {r.get('expect') or '-':<10} {r['verdict']:<11} "
            f"{r['status']:<12} {r['latency_ms']:>7.0f} {r.get('prompt_tokens') or '-':>6}  {mark(r)}"
        )
    return "\n".join(lines)


def _box(box: list | None) -> tuple[float, float, float, float] | None:
    return (box[0], box[1], box[2], box[3]) if box else None


def run(
    cases_path: str | Path,
    backend: Backend,
    out_dir: str | Path = "results",
    media: str | Path | None = None,
    only: list[str] | None = None,
    check_dirs: tuple = (),
    echo=print,
    save_images: bool = False,
    infer: str | None = None,
    infer_service: str = "rfdetr",
    boxes_from: str | Path | None = None,
) -> tuple[Path, dict]:
    """Run every case (or those whose id starts with one of ``only``), write
    ``<out_dir>/<time>-<model>.jsonl`` plus a ``.md`` summary, and return the path and summary.
    ``save_images``: also write each case's images, boxes drawn, to ``<time>-<model>/<case>/``.
    ``boxes_from``: an earlier results ``.jsonl``; each case reuses the time and box recorded
    there, so two servers are compared on exactly the same input (and no detector is needed)."""
    cases, folder = load_cases(cases_path, media)
    if only:
        cases = [c for c in cases if any(c.id.startswith(o) for o in only)]
    missing = sorted({c.clip for c in cases if not (folder / c.clip).is_file()})
    if missing:
        raise FileNotFoundError(f"clips not found in {folder}: {', '.join(missing)} (use --media to point at them)")
    reused: dict[str, dict] = {}
    if boxes_from:
        for line in Path(boxes_from).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                reused[r["case"]] = r
        absent = [c.id for c in cases if c.id not in reused]
        if absent:
            raise ValueError(f"{boxes_from} has no result for: {', '.join(absent)}")
        cases = [c.model_copy(update={"at": reused[c.id]["at"], "box": _box(reused[c.id]["box"])}) for c in cases]
    auto = any(c.box in ("auto", "auto-group") for c in cases)
    detector = Detector(infer, infer_service) if infer and auto else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = "".join(ch if ch.isalnum() else "-" for ch in backend.model).strip("-").lower()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stamp}-{slug}.jsonl"
    rows: list[dict] = []
    t0 = time.perf_counter()
    with path.open("w") as f:
        for n, case in enumerate(cases, start=1):
            shots_dir = path.with_suffix("") / case.id if save_images else None
            v, used = run_case(case, folder, backend, check_dirs, shots_dir, detector)
            if case.id in reused:
                used["box_from"] = f"{reused[case.id]['box_from']} (reused from {Path(boxes_from).name})"
            row = {
                "case": case.id,
                "clip": case.clip,
                "expect": case.expect,
                "backend": backend.url,
                **used,
                **v.model_dump(),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            mark = (
                "" if not case.expect else (" ok" if v.verdict == case.expect else f" WRONG (expected {case.expect})")
            )
            why = f"\n    {v.reason}" if v.status == "no_verdict" and v.reason else ""
            echo(f"[{n}/{len(cases)}] {case.id}: {v.verdict} ({v.status}) {v.latency_ms:.0f} ms{mark}{why}")
    if detector is not None:
        detector.close()
    summary = {**summarize(rows), "model": backend.model, "backend": backend.url,
               "wall_s": round(time.perf_counter() - t0, 1), "results": path.name}  # fmt: skip
    md = path.with_suffix(".md")
    md.write_text(
        f"# {backend.model} @ {backend.url}\n\n{stamp} UTC, {len(rows)} cases\n\n"
        f"```\n{table(rows)}\n```\n\n```json\n{json.dumps(summary, indent=2)}\n```\n"
    )
    return path, summary
