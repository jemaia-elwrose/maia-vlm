"""Events, keyframes and the images the model sees.

An event says what the detector saw (label, box, time). Frames come from a clip or from
arrays a caller already has (live frames from a ring, in 0.1.2). The check's frame plan picks
a few distinct frames around the event and cuts them into model-sized images, keeping the
geometry needed to map boxes between the source frame and each image.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict

from .checks import FramePlan

Box = tuple[float, float, float, float]  # x1, y1, x2, y2
BoxFormat = Literal["norm1000", "norm1", "pixels"]


class Event(BaseModel):
    """What the detector reported. ``box`` is in pixels of the source frame."""

    model_config = ConfigDict(extra="allow")
    label: str
    camera: str = "camera"
    box: Box | None = None
    at_s: float | None = None  # seconds into the clip; None = the middle of the clip
    zone: list[tuple[float, float]] | None = None  # polygon in source pixels, for zone checks
    clock: str = ""  # wall-clock time for the prompt, e.g. "14:02:11"


@dataclass
class Frame:
    t_s: float
    rgb: np.ndarray  # H x W x 3, uint8


@dataclass
class Shot:
    """One image sent to the model, and where it came from."""

    frame_index: int  # into the chosen frames
    t_s: float
    kind: Literal["full", "subject"]
    image: Image.Image
    x0: float  # source pixel at the image's left edge
    y0: float
    scale: float  # image pixels per source pixel
    at_event: bool = False  # taken at the event time: the only image the detector's box is known for
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def to_image(self, box: Box) -> Box:
        x1, y1, x2, y2 = box
        s = self.scale
        return ((x1 - self.x0) * s, (y1 - self.y0) * s, (x2 - self.x0) * s, (y2 - self.y0) * s)

    def jpeg_b64(self, quality: int = 90) -> str:
        buf = io.BytesIO()
        self.image.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------------- box formats
def box_to(fmt: BoxFormat, box: Box, size: tuple[int, int]) -> list[int | float]:
    """A box in image pixels, written the way the model reads and writes boxes."""
    w, h = size
    x1, y1, x2, y2 = (max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3]))
    if fmt == "pixels":
        return [round(x1), round(y1), round(x2), round(y2)]
    if fmt == "norm1":
        return [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / h, 3)]
    return [round(1000 * x1 / w), round(1000 * y1 / h), round(1000 * x2 / w), round(1000 * y2 / h)]


def box_from(fmt: BoxFormat, box: Sequence[float], size: tuple[int, int]) -> Box:
    w, h = size
    x1, y1, x2, y2 = (float(v) for v in box)
    if fmt == "norm1":
        return (x1 * w, y1 * h, x2 * w, y2 * h)
    if fmt == "norm1000":
        return (x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000)
    return (x1, y1, x2, y2)


def inside(a: Box, b: Box) -> float:
    """Share of box ``a`` that lies within box ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def grow(box: Box, by: float) -> Box:
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - by * w, box[1] - by * h, box[2] + by * w, box[3] + by * h)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------------------ reading
def read_clip(path: str | Path, start_s: float = 0.0, end_s: float | None = None) -> list[Frame]:
    """Decode the frames of a clip between two times (seconds from its first frame)."""
    import av  # imported here: listing checks or the tests of other parts don't need FFmpeg

    frames: list[Frame] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        first = stream.start_time * stream.time_base if stream.start_time is not None else 0.0
        if start_s > 1.0:
            container.seek(int((first + start_s - 1.0) / stream.time_base), stream=stream, backward=True)
        for f in container.decode(stream):
            if f.time is None:
                continue
            t = float(f.time - first)
            if t < start_s - 1e-6:
                continue
            if end_s is not None and t > end_s + 1e-6:
                break
            frames.append(Frame(t, f.to_ndarray(format="rgb24")))
    return frames


def clip_duration(path: str | Path) -> float:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration or 0) / 1e6


# ---------------------------------------------------------------------------- selection
def _thumb(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    step = max(1, max(h, w) // 64)
    return rgb[::step, ::step].mean(axis=2)


def select(frames: Sequence[Frame], at_s: float, plan: FramePlan, min_change: float = 2.0) -> list[Frame]:
    """``plan.count`` frames spread over the window around ``at_s``, the event frame always in.
    A frame almost identical to the previous pick (mean absolute difference of a small grey
    thumbnail under ``min_change``) is swapped for the next distinct one, if the window has one."""
    if not frames:
        return []
    lo, hi = at_s + plan.window_s[0], at_s + plan.window_s[1]
    pool = [f for f in frames if lo - 1e-6 <= f.t_s <= hi + 1e-6] or list(frames)
    nearest = min(pool, key=lambda f: abs(f.t_s - at_s))
    if plan.count == 1 or len(pool) == 1:
        return [nearest]
    targets = np.linspace(pool[0].t_s, pool[-1].t_s, plan.count)
    picks: list[int] = []
    for t in targets:
        i = min(range(len(pool)), key=lambda k: abs(pool[k].t_s - t))
        while i in picks and i + 1 < len(pool):
            i += 1
        if i not in picks:
            picks.append(i)
    event_i = pool.index(nearest)
    if event_i not in picks:  # replace the closest pick with the event frame itself
        closest = min(range(len(picks)), key=lambda k: abs(pool[picks[k]].t_s - at_s))
        picks[closest] = event_i
    picks.sort()
    # a pick that looks like the previous one moves forward to the first distinct frame before
    # the next pick; when there is none it stays (a still scene is still evidence)
    out = [picks[0]]
    for n, i in enumerate(picks[1:], start=1):
        limit = picks[n + 1] if n + 1 < len(picks) else len(pool)
        prev = _thumb(pool[out[-1]].rgb)
        chosen = i
        if i != event_i:
            for j in range(i, limit):
                if np.abs(_thumb(pool[j].rgb) - prev).mean() >= min_change:
                    chosen = j
                    break
        out.append(chosen)
    return [pool[i] for i in out]


# -------------------------------------------------------------------------------- shots
def _resize(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.BICUBIC)
    return img, s


def _subject_region(box: Box, context: float) -> tuple[int, int, int]:
    """A square centred on the detection: left, top, side. It may reach past the frame's edges;
    that part is filled with black, so the subject stays in the middle (a crop slid back inside
    the frame would put someone else in the middle, and the prompt says "in the middle")."""
    x1, y1, x2, y2 = box
    side = max(32.0, max(x2 - x1, y2 - y1) * (1 + 2 * context))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return round(cx - side / 2), round(cy - side / 2), round(side)


def shots(frames: Sequence[Frame], event: Event, plan: FramePlan) -> list[Shot]:
    """The images for the model: whole frames, subject crops, or one whole frame + crops.
    The detector's box is known for the event frame only; a moving subject drifts from the
    middle of the other crops, which the prompt says."""
    if plan.crop != "full" and event.box is None:
        raise ValueError("this check crops around the detection: the event needs a box")
    at = event.at_s if event.at_s is not None else frames[len(frames) // 2].t_s
    event_i = min(range(len(frames)), key=lambda k: abs(frames[k].t_s - at))
    out: list[Shot] = []

    def full(i: int, f: Frame) -> Shot:
        img, s = _resize(Image.fromarray(f.rgb), plan.max_side)
        return Shot(i, f.t_s, "full", img, 0.0, 0.0, s, at_event=i == event_i)

    def subject(i: int, f: Frame) -> Shot:
        left, top, side = _subject_region(event.box, plan.context)
        crop = Image.fromarray(f.rgb).crop((left, top, left + side, top + side))  # outside: black
        img, s = _resize(crop, plan.max_side)
        return Shot(i, f.t_s, "subject", img, left, top, s, at_event=i == event_i)

    if plan.crop == "both":
        out.append(full(event_i, frames[event_i]))
    for i, f in enumerate(frames):
        out.append(full(i, f) if plan.crop == "full" else subject(i, f))
    return out
