"""Checks: one YAML file per question to ask about an event.

A check says which events it applies to, which frames and crops to show the model, the
prompt, and the JSON Schema the answer must match. Adding a use case means adding a file.
"""

from __future__ import annotations

import hashlib
import string
from pathlib import Path
from typing import Any, Literal

import jsonschema
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VERDICTS = ("confirmed", "rejected", "unclear")

# Fields a prompt may use. They come from the event and the engine, never from the image.
PROMPT_FIELDS = {"n_frames", "time", "camera", "label", "subject", "zone", "span_s"}


class AppliesTo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[str] = Field(default_factory=lambda: ["*"])
    cameras: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("labels", "cameras", mode="before")
    @classmethod
    def _listify(cls, v: Any) -> list[str]:
        return [v] if isinstance(v, str) else list(v)

    def matches(self, label: str, camera: str) -> bool:
        return ("*" in self.labels or label in self.labels) and ("*" in self.cameras or camera in self.cameras)


class FramePlan(BaseModel):
    """Which frames the model sees, and how they are cut."""

    model_config = ConfigDict(extra="forbid")
    count: int = Field(4, ge=1, le=8)
    window_s: tuple[float, float] = (-1.0, 1.0)  # around the event time
    crop: Literal["full", "subject", "both"] = "full"
    context: float = Field(0.5, ge=0, le=3)  # subject crops grow by this much of the box on each side
    max_side: int = Field(448, ge=128, le=1536)  # long side of every image sent

    @model_validator(mode="after")
    def _window(self) -> FramePlan:
        if self.window_s[0] > 0 or self.window_s[1] < 0:
            raise ValueError("window_s must contain the event time: (before <= 0, after >= 0)")
        return self


class Grounding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # share of a cited evidence box that must lie inside the (slightly grown) detection: a head
    # inside a person box counts, a box on the other side of the frame does not
    min_inside: float = Field(0.5, ge=0, le=1)


class VerdictFrom(BaseModel):
    """Derive the verdict from a specific answer field. Small models handle "is X there?" better
    than "is the alert right?", and sometimes contradict themselves between the two; the engine
    then trusts the specific field, but never turns a contradiction into a confirmation."""

    model_config = ConfigDict(extra="forbid")
    field: str
    map: dict[str, Literal["confirmed", "rejected", "unclear"]]


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")
    version: int = 1
    title: str = ""
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    frames: FramePlan = Field(default_factory=FramePlan)
    prompt: str
    answer: dict[str, Any]  # JSON Schema of the reply
    grounding: Grounding = Field(default_factory=Grounding)
    verdict_from: VerdictFrom | None = None
    temperature: float = Field(0.0, ge=0, le=2)
    max_tokens: int = Field(256, ge=16, le=4096)
    review_only: bool = False  # temporal or sensitive checks: a "confirmed" means "needs review"
    sha256: str = ""  # of the source file, set by load()

    @field_validator("prompt")
    @classmethod
    def _fields(cls, v: str) -> str:
        used = {f for _, f, _, _ in string.Formatter().parse(v) if f}
        unknown = used - PROMPT_FIELDS
        if unknown:
            raise ValueError(f"unknown prompt fields {sorted(unknown)}; allowed: {sorted(PROMPT_FIELDS)}")
        return v

    @field_validator("answer")
    @classmethod
    def _schema(cls, v: dict) -> dict:
        jsonschema.Draft202012Validator.check_schema(v)
        props = v.get("properties", {})
        verdict = props.get("verdict", {})
        if v.get("type") != "object" or "verdict" not in v.get("required", []):
            raise ValueError("answer must be an object schema with a required 'verdict'")
        if sorted(verdict.get("enum", [])) != sorted(VERDICTS):
            raise ValueError(f"answer.verdict must be an enum of exactly {list(VERDICTS)}")
        return v

    @model_validator(mode="after")
    def _verdict_from(self) -> Check:
        if self.verdict_from is None:
            return self
        prop = self.answer.get("properties", {}).get(self.verdict_from.field)
        if not prop or "enum" not in prop:
            raise ValueError(f"verdict_from.field '{self.verdict_from.field}' must be an enum in answer")
        if self.verdict_from.field not in self.answer.get("required", []):
            raise ValueError(f"verdict_from.field '{self.verdict_from.field}' must be required in answer")
        if sorted(self.verdict_from.map) != sorted(prop["enum"]):
            raise ValueError(f"verdict_from.map must give a verdict for each of {prop['enum']}")
        return self

    def render(self, **fields: Any) -> str:
        return self.prompt.format(**{k: fields.get(k, "") for k in PROMPT_FIELDS})


def load(path: str | Path) -> Check:
    path = Path(path)
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a check is a YAML mapping")
    check = Check.model_validate(data)
    if check.id != path.stem:
        raise ValueError(f"{path}: id '{check.id}' must match the file name")
    check.sha256 = hashlib.sha256(raw).hexdigest()
    return check


def builtin_dir() -> Path:
    """The example checks: packaged with the wheel, or the repo's checks/ when installed editable."""
    packaged = Path(__file__).parent / "builtin_checks"
    return packaged if packaged.is_dir() else Path(__file__).resolve().parents[2] / "checks"


def load_all(*dirs: str | Path) -> dict[str, Check]:
    """Checks by id; later folders override earlier ones (a private folder over the built-ins)."""
    found: dict[str, Check] = {}
    for d in dirs or (builtin_dir(),):
        for path in sorted(Path(d).glob("*.yaml")):
            check = load(path)
            found[check.id] = check
    return found


def find(name: str, *dirs: str | Path) -> Check:
    """A check by id (searched in ``dirs``, then the built-ins) or by path to a YAML file."""
    if name.endswith((".yaml", ".yml")) or "/" in name:
        return load(name)
    checks = load_all(builtin_dir(), *dirs)
    if name not in checks:
        raise KeyError(f"no check '{name}'; known: {', '.join(sorted(checks))}")
    return checks[name]
