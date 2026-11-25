"""Run a list of cases (clip + check + event) and keep every verdict, for testing and timing.

cases.yaml:

    media: media                 # folder of the clips, relative to this file
    defaults: {fps: 5, at: 5}    # applied to every case unless it says otherwise
    cases:
      - {id: ppe-helmet, clip: ppe.mp4, check: helmet, box: [581, 371, 1113, 901], expect: rejected}
      - {id: shop-describe, clip: shop.mp4, check: describe}

``expect`` is optional: cases with it count towards accuracy, the others are just timed.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
    box: tuple[float, float, float, float] | None = None
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


def run_case(
    case: Case, folder: Path, backend: Backend, check_dirs: tuple = (), save_to: Path | None = None
) -> Verdict:
    check = checks_mod.find(case.check, *check_dirs)
    event = Event(label=case.label, box=case.box, at_s=case.at, zone=case.zone)
    return verify_clip(folder / case.clip, event, check, backend, sample_fps=case.fps, save_to=save_to)


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


def run(
    cases_path: str | Path,
    backend: Backend,
    out_dir: str | Path = "results",
    media: str | Path | None = None,
    only: list[str] | None = None,
    check_dirs: tuple = (),
    echo=print,
    save_images: bool = False,
) -> tuple[Path, dict]:
    """Run every case (or those whose id starts with one of ``only``), write
    ``<out_dir>/<time>-<model>.jsonl`` plus a ``.md`` summary, and return the path and summary.
    ``save_images``: also write each case's images, boxes drawn, to ``<time>-<model>/<case>/``."""
    cases, folder = load_cases(cases_path, media)
    if only:
        cases = [c for c in cases if any(c.id.startswith(o) for o in only)]
    missing = sorted({c.clip for c in cases if not (folder / c.clip).is_file()})
    if missing:
        raise FileNotFoundError(f"clips not found in {folder}: {', '.join(missing)} (use --media to point at them)")
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
            v = run_case(case, folder, backend, check_dirs, shots_dir)
            row = {
                "case": case.id,
                "clip": case.clip,
                "expect": case.expect,
                "backend": backend.url,
                **v.model_dump(),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            mark = (
                "" if not case.expect else (" ok" if v.verdict == case.expect else f" WRONG (expected {case.expect})")
            )
            echo(f"[{n}/{len(cases)}] {case.id}: {v.verdict} ({v.status}) {v.latency_ms:.0f} ms{mark}")
    summary = {**summarize(rows), "model": backend.model, "backend": backend.url,
               "wall_s": round(time.perf_counter() - t0, 1), "results": path.name}  # fmt: skip
    md = path.with_suffix(".md")
    md.write_text(
        f"# {backend.model} @ {backend.url}\n\n{stamp} UTC, {len(rows)} cases\n\n"
        f"```\n{table(rows)}\n```\n\n```json\n{json.dumps(summary, indent=2)}\n```\n"
    )
    return path, summary
