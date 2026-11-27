"""maia-vlm checks | ping | check"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import checks as checks_mod
from .backends import Backend, BackendError
from .frames import Event


def _floats(text: str, n: int | None = None) -> tuple[float, ...]:
    vals = tuple(float(v) for v in text.replace(" ", "").split(","))
    if n is not None and len(vals) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers")
    return vals


def _backend_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("backend (defaults from MAIA_VLM_* variables, else a local Ollama)")
    g.add_argument("--url", help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
    g.add_argument("--model", help="model name as the server knows it, e.g. qwen3-vl:4b-instruct")
    g.add_argument("--schema-via", choices=["response_format", "vllm", "none"])
    g.add_argument("--box-format", choices=["norm1000", "norm1", "pixels"])
    g.add_argument("--timeout", type=float, help="seconds to wait for one reply (default 120)")


def _backend(a: argparse.Namespace) -> Backend:
    return Backend.from_env(
        url=a.url, model=a.model, schema_via=a.schema_via, box_format=a.box_format, timeout_s=a.timeout
    )


def cmd_checks(a: argparse.Namespace) -> int:
    for c in checks_mod.load_all(checks_mod.builtin_dir(), *a.dir).values():
        flag = "  (review only)" if c.review_only else ""
        print(f"{c.id:<20} {c.frames.count} x {c.frames.crop:<8} {c.title}{flag}")
    return 0


def cmd_ping(a: argparse.Namespace) -> int:
    backend = _backend(a)
    try:
        models = backend.ping()
    except BackendError as exc:
        print(f"no answer: {exc}", file=sys.stderr)
        return 1
    print(f"{backend.url}: {len(models)} model(s)")
    for m in models:
        print(f"  {m}{'   <- configured' if m == backend.model else ''}")
    if backend.model not in models:
        print(f"note: '{backend.model}' is not in that list (Ollama: ollama pull {backend.model})", file=sys.stderr)
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    from .verify import verify_clip

    check = checks_mod.find(a.check, *a.dir)
    plan = {k: v for k, v in {"count": a.frames, "max_side": a.max_side}.items() if v is not None}
    if plan:  # a one-off override, e.g. to fit a small context window; the check file is unchanged
        frames = type(check.frames).model_validate({**check.frames.model_dump(), **plan})
        check = check.model_copy(update={"frames": frames})
    zone = None
    if a.zone:
        pts = _floats(a.zone.replace(";", ","))
        zone = list(zip(pts[0::2], pts[1::2], strict=True))
    event = Event(label=a.label, camera=a.camera, box=a.box, at_s=a.at, zone=zone)
    backend = _backend(a)
    verdict = verify_clip(a.clip, event, check, backend, sample_fps=a.fps, save_to=a.save_images)
    out = verdict.model_dump(exclude=set() if a.raw else {"raw"})
    print(json.dumps(out, indent=2))
    return 0 if verdict.verdict != "no_verdict" else 2


def cmd_run(a: argparse.Namespace) -> int:
    from .runner import run, table

    path, summary = run(
        a.cases,
        _backend(a),
        a.out,
        a.media,
        a.only,
        tuple(a.dir),
        save_images=a.save_images,
        infer=a.infer,
        infer_service=a.infer_service,
        boxes_from=a.boxes_from,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    print()
    print(table(rows))
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nsaved: {path} and {path.with_suffix('.md').name}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import logging
    import os
    import signal

    from .live import Rules, Service

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    def _stop(*_):  # a script's kill (SIGTERM) ends the service like Ctrl+C: summary included
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    rules = Rules.load(a.rules)
    service = Service(
        rules, _backend(a), a.redis, a.gateway, workers=a.workers, max_pending=a.max_pending,
        sample_fps=a.fps, group=a.group, consumer=a.consumer, check_dirs=tuple(a.dir),
        api_key=os.getenv("MAIA_API_KEY", ""), log_path=a.log, save_dir=a.save_images,
    )  # fmt: skip
    print(
        f"maia-vlm {__version__}: following events on {a.redis}, verdicts to {a.gateway}, model {service.backend.name}"
    )
    for r in rules.rules:
        print(f"  camera {r.camera}, label {r.label}: {', '.join(r.checks)}")
    stats = service.run()
    print(f"\nmaia-vlm {__version__} stopped")
    sources = f"{stats.from_ring} from the ring, {stats.from_clip} from clips"
    print(f"\n{stats.events} events, {stats.checks} checks ({sources}), verdicts {stats.verdicts}")
    print(f"overloaded {stats.overloaded}, errors {stats.errors}")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="maia-vlm", description="Check detector events with a vision-language model.")
    ap.add_argument("--version", action="version", version=f"maia-vlm {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("checks", help="list the available checks")
    p.add_argument("--dir", action="append", default=[], help="another folder of check YAML files")
    p.set_defaults(fn=cmd_checks)

    p = sub.add_parser("ping", help="is the backend up, and which models does it have?")
    _backend_args(p)
    p.set_defaults(fn=cmd_ping)

    p = sub.add_parser("check", help="run one check on an event in a clip")
    p.add_argument("clip", type=Path)
    p.add_argument("--check", required=True, help="check id, or a path to a check YAML")
    p.add_argument("--label", default="person", help="what the detector reported")
    p.add_argument("--box", type=lambda s: _floats(s, 4), help="x1,y1,x2,y2 in pixels of the clip")
    p.add_argument("--at", type=float, help="event time in seconds into the clip (default: the middle)")
    p.add_argument("--camera", default="camera")
    p.add_argument("--zone", help="polygon in pixels: x,y;x,y;x,y")
    p.add_argument("--dir", action="append", default=[], help="another folder of check YAML files")
    p.add_argument("--fps", type=float, help="thin the clip to this rate first, like a gateway ring (e.g. 5)")
    p.add_argument("--frames", type=int, help="override the check's number of frames (1-8)")
    p.add_argument("--max-side", type=int, help="override the check's image size, in pixels (128-1536)")
    p.add_argument("--raw", action="store_true", help="include the model's raw replies")
    p.add_argument("--save-images", type=Path, metavar="DIR", help="write the images sent, boxes drawn, to DIR")
    _backend_args(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("run", help="run a YAML list of cases, save every verdict and a timing summary")
    p.add_argument("cases", type=Path, help="cases YAML (see sets/test-clips/cases.yaml)")
    p.add_argument("--media", type=Path, help="folder of the clips (default: the cases file's 'media')")
    p.add_argument("--out", type=Path, default=Path("results"), help="where results go (default: results/)")
    p.add_argument("--only", action="append", help="run only cases whose id starts with this (repeatable)")
    p.add_argument("--dir", action="append", default=[], help="another folder of check YAML files")
    p.add_argument("--save-images", action="store_true", help="also write each case's images, boxes drawn")
    p.add_argument("--infer", help="maia-infer endpoint for cases with box: auto (e.g. tcp://localhost:5555)")
    p.add_argument("--infer-service", default="rfdetr", help="its detection worker (default rfdetr; or yolo)")
    p.add_argument("--boxes-from", type=Path, help="reuse the time and box of each case from an earlier results .jsonl")
    _backend_args(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("serve", help="follow maia-gateway's events, check them, post the verdicts to their clips")
    p.add_argument("--rules", type=Path, required=True, help="which checks run on which events (rules.yaml)")
    p.add_argument("--redis", default="redis://localhost:6379/0")
    p.add_argument(
        "--gateway", default="http://localhost:8080", help="maia-gateway's API (MAIA_API_KEY if it needs one)"
    )
    p.add_argument("--workers", type=int, default=1, help="checks run at once (vLLM batches them; Ollama: keep 1)")
    p.add_argument("--max-pending", type=int, default=32, help="queued checks before the oldest are dropped")
    p.add_argument("--fps", type=float, help="thin the frames to this rate first")
    p.add_argument("--group", default="maia-vlm", help="consumer group: services in one group split the events")
    p.add_argument("--consumer", default="vlm-1", help="this service's name in the group")
    p.add_argument("--log", type=Path, help="also append every verdict to this JSONL file")
    p.add_argument("--save-images", type=Path, metavar="DIR", help="write each check's images to DIR/<clip>/<check>/")
    p.add_argument("--dir", action="append", default=[], help="another folder of check YAML files")
    _backend_args(p)
    p.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    try:
        sys.exit(a.fn(a))
    except (KeyError, ValueError, BackendError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
