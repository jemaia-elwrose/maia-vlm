"""See what the model saw: the images of one check, with the boxes drawn on copies of them.

Green: the detector's box, on the images taken at the event time (where it is known).
Red: the box the model gave as evidence. The images sent to the model are never drawn on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import ImageDraw

from .frames import BoxFormat, Event, Shot, box_from

if TYPE_CHECKING:
    from .verify import Verdict

GREEN, RED = (40, 200, 90), (230, 50, 50)


def save_shots(folder: str | Path, images: Sequence[Shot], event: Event, verdict: Verdict, fmt: BoxFormat) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    evidence: dict[int, list] = {}
    for item in verdict.answer.get("evidence") or []:
        if isinstance(item, dict) and isinstance(item.get("image"), int) and isinstance(item.get("box"), list):
            evidence.setdefault(item["image"], []).append(item["box"])
    for n, shot in enumerate(images, start=1):
        img = shot.image.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        width = max(2, round(max(img.size) / 200))
        if event.box is not None and shot.at_event:
            draw.rectangle(shot.to_image(event.box), outline=GREEN, width=width)
        for box in evidence.get(n, []):
            if len(box) == 4:
                draw.rectangle(box_from(fmt, box, shot.size), outline=RED, width=width)
        tag = "event" if shot.at_event else f"{shot.t_s:.2f}s"
        img.save(folder / f"{n:02d}-{shot.kind}-{tag}.jpg", quality=90)
    (folder / "verdict.json").write_text(json.dumps(verdict.model_dump(), indent=2))
    return folder
