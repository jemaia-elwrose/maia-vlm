import json

import httpx
import pytest

from maia_vlm import Event, find_check, verify_clip
from maia_vlm.cli import main
from maia_vlm.guard import parse

PERSON = Event(label="person", box=(300, 120, 380, 220), at_s=2.0, camera="gate-1")


def test_a_valid_answer_becomes_a_verdict(server, backend, clip):
    server.replies = [{"verdict": "confirmed", "description": "A person stands by the gate.", "evidence": []}]
    v = verify_clip(clip, PERSON, find_check("person-present"), backend)
    assert v.verdict == "confirmed" and v.status == "ok" and v.calls == 1
    assert v.description == "A person stands by the gate."
    assert v.check == "person-present" and len(v.check_sha256) == 64 and v.model == "fake-vl@http://fake/v1"
    req = server.requests[0]
    assert req["response_format"]["json_schema"]["schema"]["required"] == ["verdict", "description"]
    assert req["temperature"] == 0.0 and req["messages"][0]["role"] == "system"
    assert len(server.images()) == 3  # crop: both, count 2 -> one whole frame + two close-ups
    assert [i["kind"] for i in v.images] == ["full", "subject", "subject"]


def test_the_prompt_carries_the_box_in_the_models_format(server, backend, clip):
    verify_clip(clip, PERSON, find_check("describe"), backend)  # crop: full, 448 px wide
    prompt = server.prompt()
    assert "box [469, 333, 594, 611]" in prompt  # 300/640*1000, 120/360*1000, ...
    assert "0-1000 scale" in prompt and "Reply with only a JSON object" in prompt
    assert "gate-1" in prompt and "2.0 s into the clip" in prompt


def test_invalid_answer_is_repaired_once(server, backend, clip):
    server.replies = [
        "Sure! The person has no helmet.",
        '```json\n{"verdict": "confirmed", "description": "Bare head, no hard hat.", '
        '"hard_hat": "no", "head_covering": "none"}\n```',
    ]
    v = verify_clip(clip, PERSON, find_check("helmet"), backend)
    assert v.verdict == "confirmed" and v.status == "repaired" and v.calls == 2
    repair = server.requests[1]["messages"]
    assert repair[-2] == {"role": "assistant", "content": "Sure! The person has no helmet."}
    assert "not valid" in repair[-1]["content"]


def test_two_bad_answers_mean_no_verdict_never_a_confirmation(server, backend, clip):
    server.replies = [{"verdict": "yes"}, {"verdict": "confirmed"}]  # wrong enum, then missing description
    v = verify_clip(clip, PERSON, find_check("helmet"), backend)
    assert v.verdict == "no_verdict" and v.status == "no_verdict" and "description" in v.reason
    assert len(v.raw) == 2


def test_backend_down_is_no_verdict(server, backend, clip):
    server.replies = [503]
    v = verify_clip(clip, PERSON, find_check("helmet"), backend)
    assert v.verdict == "no_verdict" and "HTTP 503" in v.reason


def test_evidence_inside_the_detection_is_grounded(server, backend, clip):
    # person-present: image 1 is the whole frame (448 x 252); the head sits inside the person
    head = [480, 340, 560, 420]  # norm1000, within the detection's [469, 333, 594, 611]
    server.replies = [{"verdict": "confirmed", "description": "A person.", "evidence": [{"image": 1, "box": head}]}]
    v = verify_clip(clip, PERSON, find_check("person-present"), backend)
    assert v.status == "ok" and v.verdict == "confirmed"


@pytest.mark.parametrize(
    "evidence, why",
    [
        ([{"image": 7, "box": [0, 0, 10, 10]}], "cites image 7"),
        ([{"image": 1, "box": [0, 0, 100, 100]}], "outside the detection"),
    ],
)
def test_ungrounded_evidence_lowers_the_verdict_to_unclear(server, backend, clip, evidence, why):
    server.replies = [{"verdict": "confirmed", "description": "A person.", "evidence": evidence}]
    v = verify_clip(clip, PERSON, find_check("person-present"), backend)
    assert v.status == "ungrounded" and v.verdict == "unclear" and why in v.reason


def test_review_only_checks_flag_a_confirmation_for_review(server, backend, clip):
    server.replies = [
        {
            "postures": ["standing", "falling", "lying"],
            "landed_on": "floor",
            "how_down": "fell",
            "verdict": "confirmed",
            "description": "Fell and stays down.",
            "still_down": True,
        }
    ]
    v = verify_clip(clip, PERSON, find_check("fall"), backend)
    assert v.verdict == "confirmed" and v.needs_review


def test_lying_down_on_purpose_is_not_a_fall_even_if_the_model_says_confirmed(server, backend, clip):
    server.replies = [
        {
            "postures": ["standing", "sitting", "lying"],
            "landed_on": "floor",
            "how_down": "lowered_self",
            "verdict": "confirmed",
            "description": "Lies down on the mat.",
        }
    ]
    v = verify_clip(clip, PERSON, find_check("fall"), backend)
    assert (v.verdict, v.status) == ("rejected", "inconsistent")
    assert "postures" in server.prompt() and len(server.images()) == 8


def test_a_fall_onto_a_bed_is_not_confirmed_even_if_the_model_says_it_fell(server, backend, clip):
    server.replies = [
        {
            "postures": ["standing", "falling", "lying"],
            "landed_on": "bed_or_furniture",
            "how_down": "fell",
            "verdict": "confirmed",
            "description": "Falls onto the bed.",
        }
    ]
    v = verify_clip(clip, PERSON, find_check("fall"), backend)
    assert (v.verdict, v.status) == ("rejected", "inconsistent") and "landed_on='bed_or_furniture'" in v.reason


def test_a_hard_hat_over_a_hood_rejects_the_alert(server, backend, clip):
    server.replies = [
        {
            "hard_hat": "yes",
            "head_covering": "cap_or_hood",
            "verdict": "confirmed",
            "description": "Hood with a hard hat over it.",
        }
    ]
    v = verify_clip(clip, PERSON, find_check("helmet"), backend)
    assert (v.verdict, v.status) == ("rejected", "inconsistent")


def test_vllm_gets_the_schema_its_own_way(server, backend, clip):
    backend.schema_via = "vllm"
    verify_clip(clip, PERSON, find_check("describe"), backend)
    req = server.requests[0]
    assert "response_format" not in req and req["structured_outputs"]["json"]["required"] == ["verdict", "description"]


@pytest.mark.parametrize(
    "text",
    ['{"verdict": "rejected"}', '```json\n{"verdict": "rejected"}\n```', 'Answer: {"verdict": "rejected"} done'],
)
def test_parse_accepts_common_wrappings(text):
    assert parse(text) == {"verdict": "rejected"}


def test_cli_check_prints_the_verdict(server, backend, clip, monkeypatch, capsys):
    server.replies = [
        {"verdict": "rejected", "description": "A hard hat is on.", "hard_hat": "yes", "head_covering": "none"}
    ]
    monkeypatch.setattr("maia_vlm.cli._backend", lambda a: backend)
    with pytest.raises(SystemExit) as done:
        main(["check", str(clip), "--check", "helmet", "--box", "300,120,380,220", "--at", "2"])
    out = json.loads(capsys.readouterr().out)
    assert done.value.code == 0 and out["verdict"] == "rejected" and "raw" not in out


def test_cli_lists_checks(capsys):
    with pytest.raises(SystemExit):
        main(["checks"])
    out = capsys.readouterr().out
    assert "helmet" in out and "fall" in out and "(review only)" in out


def test_verdict_reports_prompt_tokens(server, backend, clip):
    v = verify_clip(clip, PERSON, find_check("describe"), backend)
    assert v.prompt_tokens is None  # the fake server reports only total_tokens

    def counting(request):
        resp = server.handler(request)
        body = json.loads(resp.content)
        body["usage"] = {"prompt_tokens": 1234}
        return httpx.Response(200, json=body)

    backend._http._transport = httpx.MockTransport(counting)
    assert verify_clip(clip, PERSON, find_check("describe"), backend).prompt_tokens == 1234


def test_context_overflow_says_what_to_change(server, backend, clip):
    def too_long(request):
        msg = "request (4435 tokens) exceeds the available context size (4096 tokens), try increasing it"
        return httpx.Response(400, json={"error": {"message": msg}})

    backend._http._transport = httpx.MockTransport(too_long)
    v = verify_clip(clip, PERSON, find_check("describe"), backend)
    assert v.verdict == "no_verdict" and "OLLAMA_CONTEXT_LENGTH=8192" in v.reason


def test_ring_like_sampling_thins_the_clip(server, backend, clip):
    v = verify_clip(clip, PERSON, find_check("fall"), backend, sample_fps=5)  # 6 frames over 5 s
    times = [i["t_s"] for i in v.images]
    assert all(round((t - times[0]) * 5, 3) == round((t - times[0]) * 5) for t in times)  # on the 0.2 s grid


def test_cli_overrides_frames_and_size(server, backend, clip, monkeypatch, capsys):
    monkeypatch.setattr("maia_vlm.cli._backend", lambda a: backend)
    with pytest.raises(SystemExit):
        main(["check", str(clip), "--check", "describe", "--frames", "2", "--max-side", "320", "--fps", "5"])
    out = json.loads(capsys.readouterr().out)
    assert len(out["images"]) == 2 and out["images"][0]["size"] == [320, 180]
    with pytest.raises(SystemExit) as bad:
        main(["check", str(clip), "--check", "describe", "--frames", "20"])
    assert bad.value.code == 1  # validated like the check file itself


@pytest.mark.parametrize(
    "image, box, status",
    [(1, [0, 0, 100, 100], "ok"), (2, [400, 400, 600, 600], "ok"), (2, [0, 0, 100, 100], "ungrounded")],
)
def test_evidence_is_held_to_the_box_on_the_event_image_only(server, backend, clip, image, box, status):
    # helmet: 3 close-ups, the event one is image 2; the subject may have moved in images 1 and 3
    evidence = [{"image": image, "box": box}]
    server.replies = [
        {
            "hard_hat": "no",
            "head_covering": "none",
            "verdict": "confirmed",
            "description": "Bare head.",
            "evidence": evidence,
        }
    ]
    v = verify_clip(clip, PERSON, find_check("helmet"), backend)
    assert v.status == status
    prompt = server.prompt()
    assert "close-up 2 (taken at the event time)" in prompt and "at most one item, on image 2" in prompt


@pytest.mark.parametrize(
    "covering, said, final, status",
    [
        ("yes", "confirmed", "rejected", "inconsistent"),  # the 0.1.0 PPE case: trust the field
        ("no", "rejected", "unclear", "inconsistent"),  # a contradiction never becomes a confirmation
        ("no", "confirmed", "confirmed", "ok"),
        ("not_visible", "unclear", "unclear", "ok"),
    ],
)
def test_verdict_follows_the_specific_answer(server, backend, clip, covering, said, final, status):
    server.replies = [{"hard_hat": covering, "head_covering": "none", "verdict": said, "description": "x"}]
    v = verify_clip(clip, PERSON, find_check("helmet"), backend)
    assert (v.verdict, v.status) == (final, status)
    if status == "inconsistent":
        assert "contradicts hard_hat" in v.reason


def test_describe_without_a_box_looks_at_the_whole_scene(server, backend, clip):
    verify_clip(clip, Event(label="person", at_s=2.0), find_check("describe"), backend)
    assert "Focus on the scene" in server.prompt()


def test_runner_saves_every_verdict_and_a_summary(server, backend, clip, tmp_path):
    from maia_vlm.runner import run

    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "c.mp4").write_bytes(clip.read_bytes())
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        "defaults: {fps: 5, at: 2}\n"
        "cases:\n"
        "  - {id: a-helmet, clip: c.mp4, check: helmet, box: [300, 120, 380, 220], expect: rejected}\n"
        "  - {id: b-describe, clip: c.mp4, check: describe}\n"
    )
    server.replies = [
        {"hard_hat": "yes", "head_covering": "none", "verdict": "confirmed", "description": "Hard hat on."},
        {"verdict": "confirmed", "description": "A square moves right.", "activity": "moving"},
    ]
    path, summary = run(cases, backend, tmp_path / "results", echo=lambda *_: None, save_images=True)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["case"] for r in rows] == ["a-helmet", "b-describe"]
    assert summary["accuracy"] == 1.0 and summary["statuses"] == {"inconsistent": 1, "ok": 1}
    assert path.with_suffix(".md").read_text().count("a-helmet") == 1
    assert (path.with_suffix("") / "a-helmet" / "verdict.json").is_file()


def test_runner_reuses_boxes_from_an_earlier_run(server, backend, clip, tmp_path):
    from maia_vlm.runner import run

    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "c.mp4").write_bytes(clip.read_bytes())
    cases = tmp_path / "cases.yaml"
    cases.write_text("cases:\n  - {id: a, clip: c.mp4, check: helmet, box: auto, at: 1, expect: confirmed}\n")
    earlier = tmp_path / "earlier.jsonl"
    earlier.write_text(json.dumps({"case": "a", "at": 2.0, "box": [300, 120, 380, 220], "box_from": "auto"}) + "\n")
    server.replies = [{"hard_hat": "no", "head_covering": "none", "verdict": "confirmed", "description": "Bare head."}]
    path, _ = run(cases, backend, tmp_path / "r", echo=lambda *_: None, boxes_from=earlier)  # no --infer needed
    row = json.loads(path.read_text())
    assert (row["at"], row["box"], row["box_from"]) == (2.0, [300, 120, 380, 220], "auto (reused from earlier.jsonl)")


def test_runner_names_missing_clips(backend, tmp_path):
    from maia_vlm.runner import run

    cases = tmp_path / "cases.yaml"
    cases.write_text("cases:\n  - {id: x, clip: nope.mp4, check: describe}\n")
    with pytest.raises(FileNotFoundError, match="nope.mp4"):
        run(cases, backend, tmp_path / "r", echo=lambda *_: None)


def test_a_timeout_is_no_verdict_with_the_time_spent(server, backend, clip):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    backend._http._transport = httpx.MockTransport(slow)
    v = verify_clip(clip, PERSON, find_check("describe"), backend)
    assert v.verdict == "no_verdict" and "no reply within 120 s" in v.reason and "--timeout" in v.reason


def test_save_images_draws_the_boxes_on_copies(server, backend, clip, tmp_path):
    server.replies = [
        {
            "hard_hat": "no",
            "head_covering": "none",
            "verdict": "confirmed",
            "description": "Bare head.",
            "evidence": [{"image": 2, "box": [400, 400, 600, 600]}],
        }
    ]
    verify_clip(clip, PERSON, find_check("helmet"), backend, save_to=tmp_path / "shots")
    names = sorted(p.name for p in (tmp_path / "shots").iterdir())
    assert names == ["01-subject-1.00s.jpg", "02-subject-event.jpg", "03-subject-3.00s.jpg", "verdict.json"]
    sent = server.images()
    assert len(sent) == 3  # what the model got is untouched: drawing happens on copies


def test_summary_counts_alerts_the_way_an_operator_would():
    from maia_vlm.runner import mark, summarize

    rows = [
        {"expect": "confirmed", "verdict": "confirmed", "status": "ok", "calls": 1, "latency_ms": 1},
        {"expect": "confirmed", "verdict": "unclear", "status": "ok", "calls": 1, "latency_ms": 1},
        {"expect": "rejected", "verdict": "unclear", "status": "ok", "calls": 1, "latency_ms": 1},
        {"expect": "rejected", "verdict": "confirmed", "status": "ok", "calls": 1, "latency_ms": 1},
    ]
    assert [mark(r) for r in rows] == ["ok", "WRONG", "no alert", "WRONG"]
    a = summarize(rows)["alerts"]
    assert (a["caught"], a["missed"], a["false_alarms"], a["recall"], a["false_alarm_rate"]) == (1, 1, 1, 0.5, 0.5)


def test_box_auto_takes_the_detectors_box_and_records_it(server, backend, clip, tmp_path, monkeypatch):
    import maia_vlm.runner as runner

    class FakeDetector:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def box(self, clip, at, label, mode="auto"):
            if label != "person":
                return None
            return (300.0, 120.0, 380.0, 220.0) if mode == "auto" else (100.0, 50.0, 600.0, 350.0)

        def close(self):
            pass

    monkeypatch.setattr(runner, "Detector", FakeDetector)
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "c.mp4").write_bytes(clip.read_bytes())
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        "defaults: {box: auto}\n"
        "cases:\n"
        "  - {id: a, clip: c.mp4, check: person-present, expect: confirmed}\n"
        "  - {id: b, clip: c.mp4, check: person-present, label: dog, expect: rejected}\n"
        "  - {id: c, clip: c.mp4, check: fight, box: auto-group, expect: confirmed}\n"
        "  - {id: d, clip: c.mp4, check: fight, label: dog, expect: rejected}\n"
    )
    server.replies = [
        {"verdict": "confirmed", "description": "A person.", "evidence": []},
        {"verdict": "confirmed", "description": "A fight.", "people_involved": 2, "evidence": []},
        {"verdict": "rejected", "description": "Nobody fights.", "people_involved": 0, "evidence": []},
    ]
    path, summary = runner.run(cases, backend, tmp_path / "r", echo=lambda *_: None, infer="tcp://x:5555")
    a, b, c, d = [json.loads(line) for line in path.read_text().splitlines()]
    assert c["box"] == [100.0, 50.0, 600.0, 350.0] and c["box_from"] == "auto-group"
    # a whole-frame check runs on the scene when no box is found; a crop check cannot
    assert d["verdict"] == "rejected" and d["box"] is None and d["box_from"] == "auto: none found, whole scene"
    assert a["box"] == [300.0, 120.0, 380.0, 220.0] and a["box_from"] == "auto" and a["at"] == 2.0  # the middle
    assert "box [469, 333, 594, 611]" in server.prompt(0)
    assert b["verdict"] == "no_verdict" and "found no dog" in b["reason"] and "crops around it" in b["reason"]
    assert summary["alerts"]["caught"] == 2


def test_box_auto_needs_a_detector(backend, clip, tmp_path):
    from maia_vlm.runner import run

    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "c.mp4").write_bytes(clip.read_bytes())
    cases = tmp_path / "cases.yaml"
    cases.write_text("cases:\n  - {id: a, clip: c.mp4, check: person-present, box: auto}\n")
    with pytest.raises(ValueError, match="--infer"):
        run(cases, backend, tmp_path / "r", echo=lambda *_: None)
