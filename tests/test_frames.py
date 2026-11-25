import numpy as np
import pytest

from maia_vlm.checks import FramePlan
from maia_vlm.frames import Event, Frame, box_from, box_to, clip_duration, read_clip, select, shots


def test_read_clip_window(clip):
    assert clip_duration(clip) == pytest.approx(4.0, abs=0.1)
    frames = read_clip(clip, 1.0, 2.0)
    assert frames[0].t_s == pytest.approx(1.0, abs=0.05) and frames[-1].t_s == pytest.approx(2.0, abs=0.05)
    assert frames[0].rgb.shape == (360, 640, 3)


def test_select_spreads_frames_and_always_keeps_the_event_frame(clip):
    frames = read_clip(clip)
    picks = select(frames, 2.0, FramePlan(count=4, window_s=(-1.0, 1.0)))
    times = [round(f.t_s, 2) for f in picks]
    assert len(picks) == 4 and times == sorted(times)
    assert any(abs(t - 2.0) < 0.05 for t in times)
    assert times[0] == pytest.approx(1.0, abs=0.05) and times[-1] == pytest.approx(3.0, abs=0.05)


def test_select_skips_near_duplicates_when_the_window_has_change():
    still = np.zeros((48, 64, 3), np.uint8)
    moved = still.copy()
    moved[:, :32] = 200
    frames = [Frame(i * 0.1, still) for i in range(10)] + [Frame(1.0 + i * 0.1, moved) for i in range(10)]
    picks = select(frames, 0.0, FramePlan(count=2, window_s=(0.0, 1.9)))
    assert picks[0].t_s == 0.0 and picks[1].rgb is moved  # not two copies of the still frame


def test_subject_crops_keep_the_geometry(clip):
    frames = read_clip(clip, 1.9, 2.1)
    event = Event(label="person", box=(300, 120, 380, 220), at_s=2.0)
    imgs = shots(frames[:2], event, FramePlan(count=2, crop="subject", context=0.5, max_side=224))
    s = imgs[0]
    assert s.kind == "subject" and max(s.size) <= 224
    x1, y1, x2, y2 = s.to_image(event.box)
    w, h = s.size
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h  # the detection sits inside the crop
    assert abs((x1 + x2) / 2 - w / 2) < 2 and abs((y1 + y2) / 2 - h / 2) < 2  # and in its middle


def test_both_adds_one_full_frame_first(clip):
    frames = read_clip(clip, 1.9, 2.1)
    event = Event(label="person", box=(300, 120, 380, 220), at_s=2.0)
    imgs = shots(frames[:3], event, FramePlan(count=3, crop="both", max_side=320))
    assert [i.kind for i in imgs] == ["full", "subject", "subject", "subject"]
    assert imgs[0].size == (320, 180)


def test_crops_need_a_box(clip):
    with pytest.raises(ValueError, match="needs a box"):
        shots(read_clip(clip, 0, 0.1), Event(label="person"), FramePlan(crop="subject"))


@pytest.mark.parametrize("fmt", ["norm1000", "norm1", "pixels"])
def test_box_formats_round_trip(fmt):
    box = (64.0, 36.0, 320.0, 180.0)
    back = box_from(fmt, box_to(fmt, box, (640, 360)), (640, 360))
    assert back == pytest.approx(box, abs=1.0)


def test_a_subject_at_the_frame_edge_stays_in_the_middle_of_its_crop():
    # a person near the right edge of an 854 px frame: the crop reaches past the edge (black there)
    frame = Frame(5.0, np.full((480, 854, 3), 90, np.uint8))
    event = Event(label="person", box=(735, 175, 839, 328), at_s=5.0)
    s = shots([frame], event, FramePlan(count=1, crop="subject", context=0.4))[0]
    x1, y1, x2, y2 = s.to_image(event.box)
    w, h = s.size
    assert abs((x1 + x2) / 2 - w / 2) <= 1 and abs((y1 + y2) / 2 - h / 2) <= 1
    px = np.asarray(s.image)
    assert px[h // 2, -2].max() < 10 and px[h // 2, w // 2].min() > 80  # past the edge: black


def test_only_the_event_frame_is_marked(clip):
    frames = read_clip(clip, 1.0, 3.0)
    event = Event(label="person", box=(300, 120, 380, 220), at_s=2.0)
    imgs = shots(select(frames, 2.0, FramePlan(count=3)), event, FramePlan(count=3, crop="subject"))
    assert [s.at_event for s in imgs] == [False, True, False]
