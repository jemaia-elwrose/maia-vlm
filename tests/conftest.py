import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from maia_vlm import Backend


class FakeServer:
    """An OpenAI-compatible chat endpoint that answers from a script and records requests."""

    def __init__(self):
        self.replies: list = []  # str, dict (sent as JSON text) or an int HTTP status
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake-vl"}]})
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0) if self.replies else {"verdict": "unclear", "description": "no script"}
        if isinstance(reply, int):
            return httpx.Response(reply, text="boom")
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 9}})

    def images(self, n: int = -1) -> list[str]:
        content = self.requests[n]["messages"][-1]["content"]
        return [c["image_url"]["url"] for c in content if c["type"] == "image_url"]

    def prompt(self, n: int = -1) -> str:
        content = self.requests[n]["messages"][-1]["content"]
        return content[-1]["text"] if isinstance(content, list) else content


@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture
def backend(server):
    b = Backend(url="http://fake/v1", model="fake-vl", transport=httpx.MockTransport(server.handler))
    yield b
    b.close()


def make_clip(path: Path, seconds: float = 4.0, fps: int = 25, size=(640, 360)) -> Path:
    """A clip with a bright square moving left to right on a dark background: frame content
    changes every frame, so keyframe selection has something to tell apart."""
    import av

    w, h = size
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(int(seconds * fps)):
            img = np.full((h, w, 3), 30, np.uint8)
            x = int(20 + i * (w - 120) / (seconds * fps))
            img[120:220, x : x + 80] = (220, 200, 40)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("clips") / "moving.mp4")
