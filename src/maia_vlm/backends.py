"""The model behind a check: any server with an OpenAI-compatible chat endpoint.

vLLM (Linux, NVIDIA), Ollama and LM Studio (Mac, Linux), mlx-vlm's server (Mac) and hosted
APIs all accept ``POST /v1/chat/completions`` with images as data URLs. What differs is how a
JSON Schema for the reply is passed, and how the model writes boxes; both are settings here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .frames import BoxFormat, Shot

SchemaVia = Literal["response_format", "vllm", "none"]


class BackendError(RuntimeError):
    def __init__(self, message: str, elapsed_ms: float = 0.0):
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


@dataclass
class Reply:
    text: str
    latency_ms: float
    usage: dict = field(default_factory=dict)


@dataclass
class Backend:
    """``url`` is the server's base URL including ``/v1`` (e.g. http://localhost:11434/v1)."""

    url: str
    model: str
    api_key: str = ""
    schema_via: SchemaVia = "response_format"
    box_format: BoxFormat = "norm1000"  # Qwen3-VL: 0-1000 relative to the image
    timeout_s: float = 120.0  # a 7-image check on a laptop can take a minute
    name: str = ""
    transport: httpx.BaseTransport | None = None  # tests plug a fake server in here

    def __post_init__(self) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._http = httpx.Client(
            base_url=self.url.rstrip("/"), timeout=self.timeout_s, headers=headers, transport=self.transport
        )
        self.name = self.name or f"{self.model}@{self.url}"

    @classmethod
    def from_env(cls, **overrides: Any) -> Backend:
        """MAIA_VLM_URL, MAIA_VLM_MODEL, MAIA_VLM_API_KEY, MAIA_VLM_SCHEMA_VIA, MAIA_VLM_BOX_FORMAT,
        MAIA_VLM_TIMEOUT (seconds).
        Defaults: a local Ollama with Qwen3-VL 4B Instruct (not the "thinking" variant: its
        reasoning text costs time and tokens a yes/no check does not need)."""
        conf = {
            "url": os.getenv("MAIA_VLM_URL", "http://localhost:11434/v1"),
            "model": os.getenv("MAIA_VLM_MODEL", "qwen3-vl:4b-instruct"),
            "api_key": os.getenv("MAIA_VLM_API_KEY", ""),
            "schema_via": os.getenv("MAIA_VLM_SCHEMA_VIA", "response_format"),
            "box_format": os.getenv("MAIA_VLM_BOX_FORMAT", "norm1000"),
            "timeout_s": float(os.getenv("MAIA_VLM_TIMEOUT", "120")),
        }
        conf.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**conf)

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------------ requests
    def messages(self, prompt: str, images: list[Shot], system: str = "") -> list[dict]:
        content: list[dict] = []
        for n, shot in enumerate(images, start=1):
            content.append({"type": "text", "text": f"Image {n}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{shot.jpeg_b64()}"}})
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "system", "content": system}] if system else []
        return [*msgs, {"role": "user", "content": content}]

    def body(self, messages: list[dict], schema: dict | None, temperature: float, max_tokens: int) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if schema is not None and self.schema_via == "response_format":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
        elif schema is not None and self.schema_via == "vllm":
            body["structured_outputs"] = {"json": schema}
        return body

    def chat(
        self, messages: list[dict], schema: dict | None = None, temperature: float = 0.0, max_tokens: int = 256
    ) -> Reply:
        t0 = time.perf_counter()
        try:
            res = self._http.post("/chat/completions", json=self.body(messages, schema, temperature, max_tokens))
        except httpx.TimeoutException as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            raise BackendError(
                f"{self.url}: no reply within {self.timeout_s:.0f} s (a slow server or many images: raise "
                "--timeout / MAIA_VLM_TIMEOUT, or use fewer or smaller images)",
                elapsed,
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.url} unreachable: {exc}") from exc
        latency = (time.perf_counter() - t0) * 1000
        if res.status_code >= 400:
            hint = ""
            if "context" in res.text.lower():
                hint = (
                    " | the server's context window is too small for these images: raise it "
                    "(Ollama: OLLAMA_CONTEXT_LENGTH=8192, or the context slider in its settings; "
                    "vLLM: --max-model-len), or send fewer or smaller images"
                )
            raise BackendError(f"{self.url}: HTTP {res.status_code}: {res.text[:300].strip()}{hint}")
        try:
            data = res.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"{self.url}: unexpected reply: {res.text[:300]}") from exc
        return Reply(text, latency, data.get("usage") or {})

    def ping(self) -> list[str]:
        """The models the server offers (raises BackendError if it does not answer)."""
        try:
            res = self._http.get("/models")
            res.raise_for_status()
            return [m.get("id", "?") for m in res.json().get("data", [])]
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError(f"{self.url}: {exc}") from exc
