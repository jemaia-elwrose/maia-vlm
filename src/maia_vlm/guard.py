"""What stands between a model's reply and a verdict: parsing, schema validation, grounding.

An answer the engine cannot trust becomes ``no_verdict`` or ``unclear``, never a confirmation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import jsonschema

from .frames import Box, BoxFormat, Shot, box_from, grow, inside

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class ParseError(ValueError):
    pass


def parse(text: str) -> dict:
    """The JSON object in a reply. Accepts bare JSON, a fenced block, or one object with
    words around it (some models add a sentence); anything else is an error, never a guess."""
    text = text.strip()
    candidates = [text]
    fenced = _FENCE.match(text)
    if fenced:
        candidates.append(fenced.group(1))
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{") : text.rindex("}") + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ParseError(f"no JSON object in the reply: {text[:120]!r}")


def validate(obj: dict, schema: dict) -> list[str]:
    """Schema errors, most specific first; empty when the answer is valid."""
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or 'answer'}: {e.message}" for e in v.iter_errors(obj)]


def grounding(
    answer: dict, shots: Sequence[Shot], detection: Box | None, box_format: BoxFormat, min_inside: float
) -> str | None:
    """Why the answer's evidence does not match what the model was shown, or None.

    Evidence items look like ``{"image": 2, "box": [x1, y1, x2, y2]}``: ``image`` counts from 1
    over the images sent, ``box`` is in the model's box format. A cited image that was never
    sent, or a box on an image taken at the event time that lies mostly outside the detection
    (grown by 25% for loose boxes), makes the answer ungrounded. Images at other times are not
    checked: the subject may have moved, and only the event frame's box is known."""
    for item in answer.get("evidence") or []:
        if not isinstance(item, dict):
            return "evidence item is not an object"
        n = item.get("image")
        if n is not None:
            if not isinstance(n, int) or not 1 <= n <= len(shots):
                return f"cites image {n}, but {len(shots)} were shown"
        box = item.get("box")
        if box is None or detection is None or n is None:
            continue
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            return f"evidence box {box!r} is not [x1, y1, x2, y2]"
        shot = shots[n - 1]
        if not shot.at_event:  # the detector's box is only known at the event time
            continue
        cited = box_from(box_format, box, shot.size)
        share = inside(cited, grow(shot.to_image(detection), 0.25))
        if share < min_inside:
            return f"evidence box on image {n} lies mostly outside the detection ({share:.0%} inside)"
    return None


def describe_errors(errors: list[str]) -> str:
    return "; ".join(errors[:3]) + (f" (+{len(errors) - 3} more)" if len(errors) > 3 else "")


def schema_text(schema: dict) -> str:
    return json.dumps(schema, separators=(",", ":"))
