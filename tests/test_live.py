"""maia-vlm serve against a real Redis and a real maia-ring ring; the gateway and the model are fakes."""

import json
import os
import time
import uuid

import httpx
import numpy as np
import pytest

redis = pytest.importorskip("redis")
maia_ring = pytest.importorskip("maia_ring")

from maia_vlm.live import EVENTS, GatewayEvent, Rules, Service  # noqa: E402

REDIS = os.getenv("REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
def r():
    client = redis.Redis.from_url(REDIS, decode_responses=True)
    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("no Redis at " + REDIS)
    client.flushdb()
    yield client
    client.flushdb()


class FakeGateway:
    """The two gateway calls maia-vlm makes: clip metadata/video, and posting verdicts."""

    def __init__(self, clip_path=None, clip_meta=None):
        self.verdicts: list[dict] = []
        self.clip_path, self.clip_meta = clip_path, clip_meta or {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/verdicts"):
            self.verdicts.append({"path": path, **json.loads(request.content)})
            return httpx.Response(201, json={"verdicts": len(self.verdicts)})
        if path.endswith(".mp4") and self.clip_path:
            return httpx.Response(200, content=self.clip_path.read_bytes())
        if self.clip_meta:
            return httpx.Response(200, json=self.clip_meta)
        return httpx.Response(404)


RULES = Rules.model_validate(
    {
        "rules": [
            {"camera": "cam-*", "label": "person", "checks": ["person-present"]},
            {"camera": "*", "label": "person", "checks": ["describe"]},
        ]
    }  # fmt: skip
)


def service(r, backend, gw, **kw) -> Service:
    http = httpx.Client(base_url="http://gw", transport=httpx.MockTransport(gw.handler))
    return Service(RULES, backend, http=http, redis_client=r, echo=lambda *_: None, **kw)


def fill_ring(r, camera: str, at_ns: int, seconds=(-2.0, 2.0), fps=5, size=(320, 180)):
    """A ring with frames around ``at_ns`` and their entries on ``frames:<camera>``, as the gateway writes them."""
    w, h = size
    writer = maia_ring.RingWriter(f"vlmtest-{uuid.uuid4().hex[:8]}", slots=64, width=w, height=h)
    t = seconds[0]
    while t <= seconds[1] + 1e-9:
        img = np.full((h, w, 3), 40, np.uint8)
        x = int(20 + (t - seconds[0]) * 40)
        img[60:140, x : x + 40] = (30, 60, 220)  # BGR: a red square, so colour order can be checked
        ref = writer.write(img, timestamp_ns=at_ns + int(t * 1e9))
        fields = {k: str(v) for k, v in ref.to_dict().items()}
        fields["shape"] = ",".join(map(str, ref.shape))
        r.xadd(f"frames:{camera}", fields)
        t += 1 / fps
    return writer


def event_fields(camera="cam-1", at_ns=None, box=(100, 60, 140, 140), size=(320, 180), clip_start_ns=None):
    at_ns = at_ns or time.time_ns()
    meta = {"box": list(box), "frame_size": list(size), "conf": 0.9}
    return {"camera": camera, "clip_id": "20261125-120000-abcdef", "at_ns": str(at_ns), "label": "person",
            "meta": json.dumps(meta), "clip_start_ns": str(clip_start_ns or at_ns - 5_000_000_000),
            "clip_end_ns": str(at_ns + 10_000_000_000)}  # fmt: skip


def test_rules_pick_checks_by_camera_and_label():
    assert RULES.checks_for("cam-1", "person") == ["person-present", "describe"]
    assert RULES.checks_for("lobby", "person") == ["describe"]
    assert RULES.checks_for("cam-1", "car") == []


def test_unknown_checks_in_rules_are_refused(r, backend):
    bad = Rules.model_validate({"rules": [{"label": "person", "checks": ["no-such-check"]}]})
    with pytest.raises(KeyError, match="no-such-check"):
        Service(bad, backend, http=httpx.Client(), redis_client=r)


def test_frames_come_from_the_ring_by_reference(r, server, backend):
    at = time.time_ns()
    writer = fill_ring(r, "cam-1", at)
    gw = FakeGateway()
    svc = service(r, backend, gw)
    server.replies = [{"verdict": "confirmed", "description": "A person by the door.", "evidence": []}]
    svc.submit(GatewayEvent.from_fields(event_fields(at_ns=at)))
    job = [j for j in svc._jobs if j.check.id == "person-present"][0]
    v = svc.run_job(job)
    assert v.verdict == "confirmed" and gw.verdicts[0]["frames_from"] == "ring"
    assert gw.verdicts[0]["path"] == "/clips/cam-1/20261125-120000-abcdef/verdicts"
    assert gw.verdicts[0]["at_ns"] == at and gw.verdicts[0]["description"] == "A person by the door."
    assert len(server.images()) == 3  # person-present: whole frame + 2 close-ups
    assert "box [312, 333, 438, 778]" in server.prompt()  # the event's box, in the model's format
    writer.close()


def test_ring_frames_are_converted_to_rgb(r):
    from maia_ring import RingPool

    from maia_vlm.live import ring_frames

    at = time.time_ns()
    writer = fill_ring(r, "cam-1", at)
    frames = ring_frames(r, RingPool(), "cam-1", at, (-1.0, 1.0))
    assert len(frames) == 11 and frames[0].t_s == pytest.approx(-1.0)
    px = frames[5].rgb[100, :, :]
    red = px[px[:, 0] > 150]
    assert len(red) and red[0].tolist() == [220, 60, 30]  # BGR in the ring, RGB to the model
    writer.close()


def test_overwritten_frames_fall_back_to_the_clip(r, server, backend, clip):
    # nothing in the ring for this event; the clip is done. The clip (640x360) is twice the
    # ring frame size (320x180), so the box is scaled up for it.
    at = time.time_ns()
    gw = FakeGateway(clip, {"status": "done", "start_ns": at - 2_000_000_000})
    svc = service(r, backend, gw)
    server.replies = [{"verdict": "confirmed", "description": "A square.", "evidence": []}]
    svc.submit(GatewayEvent.from_fields(event_fields(at_ns=at, box=(150, 60, 190, 110))))
    job = [j for j in svc._jobs if j.check.id == "person-present"][0]
    v = svc.run_job(job)
    assert v.verdict == "confirmed" and gw.verdicts[0]["frames_from"] == "clip"
    assert "box [469, 333, 594, 611]" in server.prompt()  # (300, 120, 380, 220) of 640x360


def test_when_the_model_falls_behind_the_oldest_checks_are_dropped(r, server, backend):
    gw = FakeGateway()
    svc = service(r, backend, gw, max_pending=2)
    now = time.time_ns()
    svc.submit(GatewayEvent.from_fields(event_fields(at_ns=now - 3_000_000_000)))  # 2 checks
    svc.submit(GatewayEvent.from_fields(event_fields(at_ns=now)))  # 2 more: the 2 oldest go
    assert svc.pending() == 2 and svc.stats.overloaded == 2
    dropped = [v for v in gw.verdicts if v["reason"] == "overloaded"]
    assert len(dropped) == 2 and all(v["verdict"] == "no_verdict" for v in dropped)
    assert all(v["at_ns"] == now - 3_000_000_000 for v in dropped)


def test_serve_follows_the_events_stream_end_to_end(r, server, backend):
    at = time.time_ns() - 3_000_000_000  # long enough ago that its frame window is complete
    writer = fill_ring(r, "cam-1", at)
    gw = FakeGateway()
    svc = service(r, backend, gw)
    server.replies = [
        {"verdict": "confirmed", "description": "A person.", "evidence": []},
        {"verdict": "confirmed", "description": "One person walks right.", "activity": "walking"},
    ]
    svc.ensure_group()
    r.xadd(EVENTS, event_fields(at_ns=at))
    stats = svc.run(until=lambda: svc.stats.checks >= 2)
    assert stats.events == 1 and stats.from_ring == 2
    assert sorted(v["check"] for v in gw.verdicts) == ["describe", "person-present"]
    assert r.xpending(EVENTS, "maia-vlm")["pending"] == 0  # acknowledged
    writer.close()
