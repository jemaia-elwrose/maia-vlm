# maia-vlm

**Check detector events with a vision-language model.** A detector says "person without a
hard hat"; maia-vlm shows a small VLM a few keyframes and close-ups of that moment, asks a
structured question, and returns a validated JSON verdict: `confirmed`, `rejected` or
`unclear`, a one-line description, and evidence it can be held to.

The VLM never runs on every frame. Detectors decide when an event is worth a second look;
maia-vlm removes their false alarms and turns the rest into alerts an operator can read.

```
$ maia-vlm check gate.mp4 --check helmet --box 410,120,520,400 --at 3.2
{
  "check": "helmet",
  "verdict": "rejected",
  "description": "The person wears a white hard hat.",
  "answer": {"verdict": "rejected", "description": "...", "head_covering": "hard_hat"},
  "status": "ok",
  "latency_ms": 840.2,
  ...
}
```

Part of a set of open building blocks for video analytics:
[maia-gateway](https://github.com/jemaia-elwrose/maia-gateway) (cameras, frames, events, clips),
[maia-infer](https://github.com/jemaia-elwrose/maia-infer) (detectors behind a broker),
[maia-ring](https://github.com/jemaia-elwrose/maia-ring) (frames in shared memory).

## How it works

1. **An event** comes in: label, box, time, camera (from the gateway, or given on the command line).
2. **Keyframes**: a few distinct frames around the event, as whole frames, close-ups of the
   detection, or both. Image size, not model size, drives latency, so images are 448 px by default.
3. **A check** (one YAML file) supplies the question and the JSON Schema of the answer.
4. **The backend**: any OpenAI-compatible server. The schema is sent with the request, so
   servers that support it (vLLM, Ollama, LM Studio) cannot return malformed JSON.
5. **Guardrails**: the answer is parsed and validated anyway; an invalid one gets one repair
   try, then becomes `no_verdict`. A check can derive its verdict from a specific field
   ("what is on the head?") and catches a model that contradicts itself (`inconsistent`).
   Evidence boxes on the event frame must fall inside the detection; otherwise the verdict
   drops to `unclear` (`ungrounded`). A model's mistake never raises an alert by itself.

Every verdict records the check id, version and file hash, the model, the images used, the
latency and which guardrail fired, so any result can be traced back.

## Quick start (Mac or Linux, Ollama)

```bash
ollama pull qwen3-vl:4b-instruct        # Ollama 0.12.7 or newer; https://ollama.com
pip install "maia-vlm @ git+https://github.com/jemaia-elwrose/maia-vlm"
maia-vlm ping                           # is the backend up, is the model there?
maia-vlm checks                         # what can be asked
maia-vlm check clip.mp4 --check describe --at 2.5
```

**Context window.** Images take most of a request: with Ollama's Qwen3-VL a 448 px image costs
about 1,000 tokens, so a 4-frame check needs about 4,500 and a 7-image one about 8,000. Ollama
defaults to a 4,096-token context and its OpenAI-compatible endpoint cannot raise it per request:
set it on the server, `OLLAMA_CONTEXT_LENGTH=16384 ollama serve` (or the context slider in the
app's settings). Every verdict reports `prompt_tokens`; `--frames` and `--max-side` shrink a
check for one run.

Use the `-instruct` models: the "thinking" variants write their reasoning before the answer,
which costs seconds and tokens a yes/no check does not need. `qwen3-vl:2b-instruct` is faster,
`qwen3-vl:8b-instruct` more accurate.

## NVIDIA GPU (vLLM)

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct --dtype half --max-model-len 8192 \
     --limit-mm-per-prompt '{"image": 8}'
export MAIA_VLM_URL=http://localhost:8000/v1 MAIA_VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct
maia-vlm check clip.mp4 --check person-present --box 410,120,520,400
```

`--dtype half` matters on a Tesla T4 and other pre-Ampere GPUs: they have no bfloat16.

Any OpenAI-compatible server works (LM Studio, mlx-vlm's server, a hosted API) through
`MAIA_VLM_URL`, `MAIA_VLM_MODEL` and `MAIA_VLM_API_KEY`. `MAIA_VLM_BOX_FORMAT` says how the model
writes boxes: `norm1000` (Qwen3-VL, the default), `norm1` or `pixels`.

## Checks

| Check | Question | Shows the model |
|---|---|---|
| `person-present` | Is there really a person where the detector says? | whole frame + close-ups |
| `helmet` | Is the person wearing a hard hat? | close-ups |
| `describe` | One-line description of what happens | 4 whole frames |
| `phone-use` | Is the person using a phone? | close-ups |
| `smoking` | Is the person smoking? | close-ups |
| `fall` | Did the person fall, and are they still down? | 6 frames over 5 s |
| `fight` | Are people fighting, or just close together? | 6 frames |
| `concealment` | Did the person put an item into a bag or clothing? | whole frame + close-ups |
| `forklift-proximity` | Is a person on foot within reach of a moving forklift? | 3 whole frames |

`fall`, `fight` and `concealment` are **review-only**: a `confirmed` sets `needs_review`, a
prompt for a person to watch the clip, never a finding on its own. Their wording describes
movements, never people.

A check is a YAML file; your own go in a folder passed with `--dir` (and override built-ins of
the same id):

```yaml
id: helmet
version: 1
applies_to: {labels: [person]}
frames: {count: 3, window_s: [-1.0, 1.0], crop: subject, context: 0.4}
prompt: |
  These are close-ups of {subject} at a work site.
  Is that person wearing a hard hat? ...
answer:                  # JSON Schema; validated, never trusted
  type: object
  required: [verdict, description]
  properties:
    verdict: {enum: [confirmed, rejected, unclear]}
    description: {type: string, maxLength: 160}
    head_covering: {enum: [hard_hat, cap_or_hood, none, not_visible]}
```

Prompts may use `{subject}`, `{label}`, `{camera}`, `{time}`, `{n_frames}`, `{span_s}` and
`{zone}`, all filled from the event; nothing from the image is pasted into a prompt.

## Live: checking maia-gateway's events

`maia-vlm serve` follows the gateway's `events` stream. For each event, a rules file says which
checks run; each check waits for its frame window to pass, reads the frames around the event
from the camera's ring by reference (falling back to the event's clip if they were already
overwritten), and posts its verdict to the clip:

```bash
pip install -e ".[live]"
maia-vlm serve --rules rules.yaml --redis redis://localhost:6379/0 --gateway http://localhost:8080
```

```yaml
# rules.yaml: a camera or label may be a glob; matching rules add up
rules:
  - {camera: "*", label: person, checks: [person-present, describe]}
  - {camera: "ppe-*", label: person, checks: [helmet]}
```

```
14:02:11 gate-1 person -> person-present: confirmed (ok, 2.4 s, frames from ring)
    A man in a dark jacket walks towards the gate.
```

The event's box (`meta.box`, in pixels of the ring frame) is what the check looks at:
maia-gateway's `examples/detect_from_bus.py` sends YOLO's most confident box. Verdicts annotate
the clip (`GET /clips/<camera>/<clip>` shows them); they never remove an alert. Services in one
consumer group (`--group`) split the events; when the model falls behind, the oldest queued
checks are answered `no_verdict` (overloaded), so detection never waits.

**One-command live test**, logs kept: `scripts/live-test.sh` checks that the gateway, the camera
and the VLM server are up, starts maia-infer with a YOLO worker if none is running, runs
`maia-vlm serve` and the detector for a while, waits for the verdicts and the clips, and writes
a summary, every log, and the images each check saw to `logs/live-<time>-<camera>/`:

```bash
CAMERA=demo-video DURATION=60 MAIA_VLM_URL=http://localhost:8001/v1 GATEWAY_DIR=../maia-gateway scripts/live-test.sh
```

## Running a set of cases

`maia-vlm run` takes a YAML list of cases (clip, check, box, time, expected verdict), runs them
all, and saves every verdict to `results/<time>-<model>.jsonl` with a Markdown summary:
accuracy on the cases with an expected verdict, statuses, latency p50/p95, prompt tokens.

```bash
maia-vlm run sets/test-clips/cases.yaml --media ~/Test-Videos
maia-vlm run sets/test-clips/cases.yaml --media ~/Test-Videos --only ppe --only gate   # a few cases
maia-vlm run sets/test-clips/cases.yaml --media ~/Test-Videos --save-images            # + what the model saw
```

`--save-images` (also on `maia-vlm check`) writes the images each case sent, with the detector's
box in green on the event image and the model's evidence box in red: the quickest way to see
why an answer was `ungrounded`, or whether a hand-made box matches the event time.

Besides exact accuracy the summary reports alerts the way an operator sees them: real events
caught or missed, and false alarms. An `unclear` on an event that should be rejected raises no
alert, so it is marked `no alert` rather than wrong.

First measurements, the same 4-frame `describe` check on the same clips:

| Machine | Server, model | Prompt tokens | Latency |
|---|---|---|---|
| Tesla T4 16 GB | vLLM, Qwen3-VL-4B-Instruct (fp16) | 737 | 2-3 s |
| Apple M4 (MacBook Air, 16 GB) | Ollama, qwen3-vl:4b-instruct | 4,457 | 27-38 s |

On the full public-v1 set (46 cases, five checks; [results](sets/public-v1/RESULTS.md)):

| Machine | Server, model | Correct | False alarms | Latency p50 |
|---|---|---|---|---|
| Tesla T4 16 GB | vLLM, Qwen3-VL-4B-Instruct (fp16) | 43 / 46 | 2 / 23 | 5.1 s |
| Apple M4 (MacBook Air, 16 GB) | mlx-vlm, Qwen3-VL-4B-Instruct (8-bit) | 41 / 46 | 4 / 23 | 10.5 s |

Checks with close-ups and evidence took 7-12 s on the T4: most of that was the model writing an
evidence box per image, so evidence is now one box, on the event image. On a laptop, slow 6-7
image checks need `--timeout` above the default 120 s.

The Ollama gap was mostly the server, not the chip: it turns each 448 px image into about six
times as many tokens as vLLM does. mlx-vlm uses the same number as vLLM, and on a Mac it is the
server to use: `scripts/mlx-run.sh` starts it, waits for the model to load and runs a set.

## Testing on maia-gateway clips

A gateway clip starts `pre_s` (default 5 s) before its event, and live checks will get frames
from a ring filled at the camera's sampling rate (default 5 fps). Both are reproducible on any
clip the gateway saved:

```bash
maia-vlm check clips/demo-video/20261124-101639-049616.mp4 --check describe --at 5 --fps 5
```

`--at 5` puts the event where the gateway put it; `--fps 5` thins the frames to what the ring
holds, so the check sees exactly what it will see live.

## As a library

```python
from maia_vlm import Backend, Event, find_check, verify_clip

event = Event(label="person", box=(410, 120, 520, 400), at_s=3.2, camera="gate-1")
verdict = verify_clip("clip.mp4", event, find_check("helmet"), Backend.from_env())
if verdict.verdict == "confirmed":
    ...
```

`verify(event, frames, check, backend)` takes frames you already have (`Frame(t_s, rgb)`), for
example from a maia-ring ring.

## Roadmap

- **0.1.2** eval (done): public-v1, 46 cases with deliberate negatives; vLLM on a T4 vs mlx-vlm on an M4.
- **0.2** fine-tuning: a QLoRA recipe for a 2-4B model, measured with the same harness.

## Development

```bash
pip install -e ".[dev]"
pytest -q          # a fake OpenAI-compatible server stands in for the model: no GPU needed
ruff check . && ruff format --check .
```

## License

Apache-2.0
