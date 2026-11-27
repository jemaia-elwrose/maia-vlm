import pytest

from maia_vlm import checks


def test_every_builtin_check_loads_and_renders():
    found = checks.load_all()
    assert set(found) == {
        "person-present", "helmet", "describe", "phone-use", "smoking",
        "fall", "fight", "concealment", "forklift-proximity",
    }  # fmt: skip
    for c in found.values():
        text = c.render(n_frames=3, subject="the person", label="person", span_s="2.0", camera="c", time="t")
        assert "{" not in text and len(c.sha256) == 64
    assert found["fall"].review_only and not found["helmet"].review_only


def write(tmp_path, name, body):
    p = tmp_path / f"{name}.yaml"
    p.write_text(body)
    return p


BASE = """id: {id}
prompt: "Is {{label}} there?"
answer:
  type: object
  required: [verdict]
  properties:
    verdict: {{enum: [confirmed, rejected, unclear]}}
"""


def test_a_private_folder_overrides_and_adds(tmp_path):
    write(tmp_path, "helmet", BASE.format(id="helmet"))
    write(tmp_path, "tailgating", BASE.format(id="tailgating"))
    found = checks.load_all(checks.builtin_dir(), tmp_path)
    assert found["helmet"].prompt == "Is {label} there?" and "tailgating" in found
    assert checks.find("tailgating", tmp_path).id == "tailgating"


@pytest.mark.parametrize(
    "body, message",
    [
        (BASE.format(id="x").replace("{label}", "{pixels}"), "unknown prompt fields"),
        (BASE.format(id="x").replace("unclear", "maybe"), "enum of exactly"),
        (BASE.format(id="x").replace("required: [verdict]", "required: []"), "required 'verdict'"),
        (BASE.format(id="x") + "frames: {window_s: [0.5, 1.0]}\n", "must contain the event time"),
        (BASE.format(id="x") + "colour: red\n", "Extra inputs"),
    ],
)
def test_bad_checks_are_refused_with_a_reason(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        checks.load(write(tmp_path, "x", body))


def test_id_must_match_the_file_name(tmp_path):
    with pytest.raises(ValueError, match="must match the file name"):
        checks.load(write(tmp_path, "y", BASE.format(id="x")))


def test_unknown_check_lists_the_known_ones():
    with pytest.raises(KeyError, match="helmet"):
        checks.find("no-such-check")


def test_verdict_from_must_cover_every_value(tmp_path):
    body = (
        BASE.format(id="x")
        .replace("properties:\n", "properties:\n    covering: {enum: [hat, none]}\n")
        .replace("required: [verdict]", "required: [covering, verdict]")
    )
    ok = body + "verdict_from: {field: covering, map: {hat: rejected, none: confirmed}}\n"
    assert checks.load(write(tmp_path, "x", ok)).verdict_from.field == "covering"
    bad = body + "verdict_from: {field: covering, map: {hat: rejected}}\n"
    with pytest.raises(ValueError, match="each of"):
        checks.load(write(tmp_path, "x", bad))


def test_the_shipped_test_cases_load():
    from maia_vlm.runner import load_cases

    cases, _ = load_cases(checks.builtin_dir().parent / "sets" / "test-clips" / "cases.yaml")
    known = checks.load_all()
    assert len(cases) >= 20 and all(c.check in known for c in cases)
    assert any(c.expect == "rejected" for c in cases)


def test_the_public_set_loads_and_is_balanced():
    from maia_vlm.runner import load_cases

    cases, _ = load_cases(checks.builtin_dir().parent / "sets" / "public-v1" / "cases.yaml")
    known = checks.load_all()
    assert all(c.check in known for c in cases)
    assert all(c.box in ("auto", "auto-group") or isinstance(c.box, tuple) for c in cases)  # a box for every case
    for check in {c.check for c in cases}:
        mine = [c.expect for c in cases if c.check == check]
        assert mine.count("confirmed") == mine.count("rejected"), check


def test_version_is_written_once():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text and 'path = "src/maia_vlm/__init__.py"' in text
