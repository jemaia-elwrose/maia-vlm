"""Live: follow maia-gateway's events, check them, write the verdicts back.

    maia-vlm serve --rules rules.yaml --redis redis://localhost:6379/0 --gateway http://localhost:8080

For each event on the gateway's ``events`` stream, the rules say which checks to run. A check
waits until the end of its frame window, takes the frames around the event from the camera's
ring by reference (``frames:<camera>``), and falls back to the event's clip when they have
already been overwritten. The verdict is posted to the clip (``POST /clips/.../verdicts``).

Detection never waits on this: when the model falls behind, the oldest checks are answered
``no_verdict`` (overloaded) instead of piling up.

rules.yaml:

    rules:
      - {camera: demo-video, label: person, checks: [person-present, describe]}
      - {camera: "*", label: fall, checks: [fall]}
"""

from __future__ import annotations

import fnmatch
import heapq
import json
import logging
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from . import checks as checks_mod
from .backends import Backend
from .checks import Check
from .frames import Event, Frame, read_clip
from .verify import Verdict, thin, verify

log = logging.getLogger("maia_vlm.live")

EVENTS = "events"
FRAMES = "frames:{}"
GROUP = "maia-vlm"


# ----------------------------------------------------------------------------- rules
class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera: str = "*"  # a name or a glob
    label: str = "*"
    checks: list[str] = Field(min_length=1)


class Rules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules: list[Rule]

    @classmethod
    def load(cls, path: str | Path) -> Rules:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def checks_for(self, camera: str, label: str) -> list[str]:
        out: list[str] = []
        for r in self.rules:
            if fnmatch.fnmatchcase(camera, r.camera) and fnmatch.fnmatchcase(label, r.label):
                out += [c for c in r.checks if c not in out]
        return out


# ----------------------------------------------------------------------------- events
@dataclass
class GatewayEvent:
    """An entry of the gateway's ``events`` stream."""

    camera: str
    clip_id: str
    at_ns: int
    label: str
    meta: dict
    clip_start_ns: int
    clip_end_ns: int

    @classmethod
    def from_fields(cls, f: dict[str, str]) -> GatewayEvent:
        return cls(
            camera=f["camera"],
            clip_id=f["clip_id"],
            at_ns=int(f["at_ns"]),
            label=f.get("label", ""),
            meta=json.loads(f.get("meta") or "{}"),
            clip_start_ns=int(f.get("clip_start_ns") or 0),
            clip_end_ns=int(f.get("clip_end_ns") or 0),
        )

    @property
    def box(self) -> tuple[float, float, float, float] | None:
        b = self.meta.get("box")
        return tuple(float(v) for v in b) if isinstance(b, list) and len(b) == 4 else None  # type: ignore[return-value]

    def clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.at_ns / 1e9))


@dataclass(order=True)
class Job:
    due: float  # wall clock, seconds: when the frames after the event exist
    seq: int
    event: GatewayEvent = field(compare=False)
    check: Check = field(compare=False)


# ----------------------------------------------------------------------------- frames
def _rgb(array: np.ndarray, pixfmt: str) -> np.ndarray:
    if pixfmt == "gray" or array.ndim == 2:
        a = array if array.ndim == 2 else array[..., 0]
        return np.repeat(a[..., None], 3, axis=2)
    if pixfmt == "bgr":
        return np.ascontiguousarray(array[..., ::-1])
    return np.ascontiguousarray(array[..., :3])


def ring_frames(r, pool, camera: str, at_ns: int, window: tuple[float, float]) -> list[Frame]:
    """Frames of ``camera`` within ``window`` seconds of ``at_ns``, read from its ring (copied,
    so they stay valid). ``t_s`` is relative to the event. Frames already overwritten are left out."""
    from maia_ring import FrameRef, StaleFrame

    lo_ns, hi_ns = at_ns + int(window[0] * 1e9), at_ns + int(window[1] * 1e9)
    # stream ids are the milliseconds the entries were added: a little after each frame's time
    entries = r.xrange(FRAMES.format(camera), min=f"{max(0, lo_ns // 1_000_000 - 2000)}-0", max="+")
    out: list[Frame] = []
    for _, f in entries:
        ts = int(f["timestamp_ns"])
        if ts < lo_ns or ts > hi_ns:
            continue
        ref = FrameRef(
            ring=f["ring"], instance=int(f["instance"]), slot=int(f["slot"]), frame_index=int(f["frame_index"]),
            seq=int(f["seq"]), shape=tuple(int(x) for x in f["shape"].split(",")), pixfmt=f["pixfmt"],
            timestamp_ns=ts, dtype=f.get("dtype", "uint8"),
        )  # fmt: skip
        try:
            frame = pool.get(ref, copy=True)
        except (StaleFrame, FileNotFoundError, OSError):
            continue
        out.append(Frame((ts - at_ns) / 1e9, _rgb(frame.array, ref.pixfmt)))
    return out


def enough(frames: list[Frame], check: Check) -> bool:
    """The ring still holds what the check needs: the event frame, and most of the window."""
    if not frames or min(abs(f.t_s) for f in frames) > 0.5:
        return False
    lo, hi = check.frames.window_s
    return frames[0].t_s <= lo + 0.5 and frames[-1].t_s >= hi - 0.5


# ----------------------------------------------------------------------------- service
@dataclass
class Stats:
    events: int = 0
    checks: int = 0
    from_ring: int = 0
    from_clip: int = 0
    overloaded: int = 0
    errors: int = 0
    verdicts: dict = field(default_factory=dict)


class Service:
    def __init__(
        self,
        rules: Rules,
        backend: Backend,
        redis_url: str = "redis://localhost:6379/0",
        gateway: str = "http://localhost:8080",
        *,
        workers: int = 1,
        max_pending: int = 32,
        sample_fps: float | None = None,
        clip_wait_s: float = 30.0,
        group: str = GROUP,
        consumer: str = "vlm-1",
        check_dirs: tuple = (),
        api_key: str = "",
        log_path: str | Path | None = None,
        save_dir: str | Path | None = None,
        echo: Callable[[str], None] = print,
        http: httpx.Client | None = None,
        redis_client=None,
    ) -> None:
        import redis
        from maia_ring import RingPool

        self.rules, self.backend = rules, backend
        self.r = redis_client or redis.Redis.from_url(redis_url, decode_responses=True)
        self.pool = RingPool()
        headers = {"X-API-Key": api_key} if api_key else {}
        self.http = http or httpx.Client(base_url=gateway, timeout=10, headers=headers)
        self.workers, self.max_pending, self.sample_fps = workers, max_pending, sample_fps
        self.clip_wait_s, self.group, self.consumer = clip_wait_s, group, consumer
        self.checks = checks_mod.load_all(checks_mod.builtin_dir(), *check_dirs)
        unknown = {c for r in rules.rules for c in r.checks} - set(self.checks)
        if unknown:
            raise KeyError(f"rules name unknown checks: {sorted(unknown)}; known: {sorted(self.checks)}")
        self.log_path = Path(log_path) if log_path else None
        self.save_dir = Path(save_dir) if save_dir else None
        self.echo = echo
        self.stats = Stats()
        self._jobs: list[Job] = []
        self._seq = 0
        self._cv = threading.Condition()
        self._stop = threading.Event()

    # ---------------------------------------------------------------- intake
    def ensure_group(self) -> None:
        import redis

        try:
            self.r.xgroup_create(EVENTS, self.group, id="$", mkstream=True)
        except redis.ResponseError as exc:  # BUSYGROUP: it exists already
            if "BUSYGROUP" not in str(exc):
                raise

    def submit(self, event: GatewayEvent) -> int:
        """Queue the checks the rules give for this event; returns how many."""
        names = self.rules.checks_for(event.camera, event.label)
        self.stats.events += 1
        dropped: list[Job] = []
        with self._cv:
            for name in names:
                check = self.checks[name]
                due = event.at_ns / 1e9 + max(0.0, check.frames.window_s[1]) + 0.3
                self._seq += 1
                heapq.heappush(self._jobs, Job(due, self._seq, event, check))
            while len(self._jobs) > self.max_pending:  # the oldest go: detection never waits on us
                oldest = min(self._jobs, key=lambda j: j.event.at_ns)
                self._jobs.remove(oldest)
                dropped.append(oldest)
            heapq.heapify(self._jobs)
            self._cv.notify_all()
        for job in dropped:
            self.stats.overloaded += 1
            base = {"check": job.check.id, "check_version": job.check.version, "check_sha256": job.check.sha256,
                    "model": self.backend.name}  # fmt: skip
            self._report(job, Verdict(**base, verdict="no_verdict", status="no_verdict", reason="overloaded"), "-")
        return len(names)

    def poll(self, block_ms: int = 1000) -> int:
        """Read new events once (consumer group: several services split them)."""
        batch = self.r.xreadgroup(self.group, self.consumer, {EVENTS: ">"}, count=50, block=block_ms)
        n = 0
        for _, entries in batch or []:
            for entry_id, fields in entries:
                try:
                    n += self.submit(GatewayEvent.from_fields(fields))
                except (KeyError, ValueError) as exc:
                    log.warning("bad event %s: %s", entry_id, exc)
                self.r.xack(EVENTS, self.group, entry_id)
        return n

    # ---------------------------------------------------------------- work
    def next_job(self, timeout: float = 0.5) -> Job | None:
        with self._cv:
            end = time.time() + timeout
            while not self._stop.is_set():
                if self._jobs and self._jobs[0].due <= time.time():
                    return heapq.heappop(self._jobs)
                wait = (self._jobs[0].due - time.time()) if self._jobs else timeout
                if time.time() >= end:
                    return None
                self._cv.wait(max(0.01, min(wait, end - time.time())))
        return None

    def frames_for(self, event: GatewayEvent, check: Check) -> tuple[list[Frame], Event, str]:
        """Frames, the event as maia-vlm sees it, and where the frames came from."""
        frames = ring_frames(self.r, self.pool, event.camera, event.at_ns, check.frames.window_s)
        if self.sample_fps:
            frames = thin(frames, self.sample_fps)
        ev = Event(label=event.label, camera=event.camera, box=event.box, at_s=0.0, clock=event.clock())
        if enough(frames, check):
            return frames, ev, "ring"
        return self.clip_frames(event, check, ev)

    def clip_frames(self, event: GatewayEvent, check: Check, ev: Event) -> tuple[list[Frame], Event, str]:
        """The ring moved on: wait for the event's clip, download it, read the window from it."""
        url = f"/clips/{event.camera}/{event.clip_id}"
        deadline = time.time() + self.clip_wait_s + max(0.0, event.clip_end_ns / 1e9 - time.time())
        while True:
            res = self.http.get(url)
            if res.status_code == 200 and res.json().get("status") == "done":
                break
            if res.status_code == 200 and res.json().get("status") == "failed":
                return [], ev, "clip failed"
            if time.time() > deadline or self._stop.is_set():
                return [], ev, "clip not ready"
            time.sleep(1.0)
        meta = res.json()
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            video = self.http.get(url + ".mp4")
            video.raise_for_status()
            tmp.write(video.content)
            tmp.flush()
            start = meta.get("start_ns") or event.clip_start_ns
            at_s = (event.at_ns - start) / 1e9
            lo, hi = check.frames.window_s
            frames = read_clip(tmp.name, max(0.0, at_s + lo), at_s + hi)
        if self.sample_fps:
            frames = thin(frames, self.sample_fps)
        box = event.box
        size = event.meta.get("frame_size")
        if box and frames and isinstance(size, list) and len(size) == 2:  # ring frame -> clip pixels
            h, w = frames[0].rgb.shape[:2]
            sx, sy = w / float(size[0]), h / float(size[1])
            box = (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)
        return frames, ev.model_copy(update={"at_s": at_s, "box": box}), "clip"

    def run_job(self, job: Job) -> Verdict:
        try:
            frames, ev, source = self.frames_for(job.event, job.check)
        except Exception as exc:  # a bad clip or a gone ring must not stop the service
            log.exception("frames for %s", job.event.clip_id)
            frames, ev, source = [], Event(label=job.event.label), f"error: {exc}"
        if source == "ring":
            self.stats.from_ring += 1
        elif source == "clip":
            self.stats.from_clip += 1
        if not frames:
            base = {"check": job.check.id, "check_version": job.check.version, "check_sha256": job.check.sha256,
                    "model": self.backend.name}  # fmt: skip
            v = Verdict(**base, verdict="no_verdict", status="no_verdict", reason=f"no frames ({source})")
        else:
            save = self.save_dir / job.event.clip_id / job.check.id if self.save_dir else None
            v = verify(ev, frames, job.check, self.backend, save_to=save)
        self._report(job, v, source)
        return v

    def _report(self, job: Job, v: Verdict, source: str) -> None:
        self.stats.checks += 1
        self.stats.verdicts[v.verdict] = self.stats.verdicts.get(v.verdict, 0) + 1
        record: dict[str, Any] = {
            "check": v.check, "verdict": v.verdict, "at_ns": job.event.at_ns, "description": v.description,
            "status": v.status, "reason": v.reason, "needs_review": v.needs_review, "model": v.model,
            "latency_ms": v.latency_ms, "prompt_tokens": v.prompt_tokens, "frames_from": source,
            "check_version": v.check_version, "answer": v.answer,
        }  # fmt: skip
        try:
            self.http.post(f"/clips/{job.event.camera}/{job.event.clip_id}/verdicts", json=record).raise_for_status()
        except httpx.HTTPError as exc:
            self.stats.errors += 1
            log.warning("could not post the verdict for %s: %s", job.event.clip_id, exc)
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"camera": job.event.camera, "clip_id": job.event.clip_id, **record}) + "\n")
        flag = "  NEEDS REVIEW" if v.needs_review else ""
        self.echo(
            f"{job.event.clock()} {job.event.camera} {job.event.label} -> {v.check}: {v.verdict} "
            f"({v.status}, {v.latency_ms / 1000:.1f} s, frames from {source}){flag}"
            + (f"\n    {v.description}" if v.description else "")
            + (f"\n    reason: {v.reason}" if v.reason else "")
        )

    def _work(self) -> None:
        while not self._stop.is_set():
            job = self.next_job()
            if job is not None:
                try:
                    self.run_job(job)
                except Exception:
                    self.stats.errors += 1
                    log.exception("check %s on %s", job.check.id, job.event.clip_id)

    # ---------------------------------------------------------------- lifecycle
    def run(self, until: Callable[[], bool] | None = None) -> Stats:
        """Follow events until Ctrl+C (or ``until()`` is true); returns the stats."""
        self.ensure_group()
        threads = [threading.Thread(target=self._work, daemon=True, name=f"vlm-{i}") for i in range(self.workers)]
        for t in threads:
            t.start()
        try:
            while not (until and until()):
                self.poll(block_ms=500)
        except KeyboardInterrupt:
            log.info("stopping: %d checks still queued are dropped", self.pending())
        finally:
            self.stop()
            for t in threads:
                t.join(timeout=5)
        return self.stats

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def pending(self) -> int:
        with self._cv:
            return len(self._jobs)
