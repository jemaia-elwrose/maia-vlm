# Changelog

## 0.1.2 (2026-11-27)
- public-v1 results ([sets/public-v1/RESULTS.md](sets/public-v1/RESULTS.md)): Tesla T4 + vLLM
  43/46, 22/23 events caught, 2 false alarms, 5.1 s p50; Apple M4 + mlx-vlm (8-bit) 41/46,
  10.5 s p50, same prompt tokens.
- `box: auto` in case files: the box comes from a maia-infer YOLO worker at the case time
  (`maia-vlm run --infer tcp://...`): the largest person, searched up to 0.8 s around the
  moment; `auto-group` boxes everyone (fights). Whole-frame checks run without a box when none
  is found. Each result records the box and time used.
- `verdict_from` can list several fields: the verdict is confirmed only if every one of them
  says confirmed.
- `helmet` v4: asks `hard_hat` yes/no first, then what else is on the head; a hard hat worn
  over a hood counts as a hard hat (public-v1 run 2: a hood under the hat read as "no hard hat").
- `fall` v4: 8 frames 0.5 s apart; the model lists the person's posture per image, says where
  they ended up (`landed_on`: floor, bed or furniture, never down) and how they got down
  (`fell`, `lowered_self`, ...). Confirmed only if both say fall. Public-v1 runs 1 and 2:
  sitting or lying down on a bed and crouching were confirmed as falls.
- `sets/public-v1`: 46 cases over five checks, real events and close-call negatives, from
  public datasets (Le2i, RLVS, Retail Shoplifting, HMDB51, PPE videos); clips not included.
- `maia-vlm run` prints the reason under each `no_verdict`.
- The version is written once, in `maia_vlm/__init__.py`; `pyproject.toml` reads it from there
  (`--version` said 0.1.1 after installing 0.1.2).
- `maia-vlm run --boxes-from <earlier .jsonl>`: reuses each case's time and box, so two
  servers are compared on the same input without a detector.
- `scripts/mlx-run.sh`: public-v1 on an mlx-vlm server (Apple Silicon), boxes reused from a
  T4 run; logs, system info and results in `logs/mlx-<time>/`.
- `scripts/prepare-clip.sh`: cuts a test clip with its moment in the middle.
- CI: macOS 15 runners (GitHub retired macOS 14).

## 0.1.1
- `maia-vlm serve --rules rules.yaml`: follows maia-gateway's `events` stream (consumer group),
  runs the checks the rules give, frames by reference from the camera's ring (BGR/gray
  converted) or, once overwritten, from the event's clip (box scaled to the clip), and posts
  each verdict to the clip. A bounded queue answers the oldest checks `no_verdict`
  (overloaded) when the model falls behind. `--log` (JSONL), `--save-images`, `--workers`.
- `scripts/live-test.sh`: the whole chain in one command, every log kept, a summary at the end.
- Needs maia-gateway 0.1.5 (events stream, verdicts on clips). `pip install "maia-vlm[live]"`.

## 0.1.0
- Checks as YAML: question, frames and crops, JSON Schema of the answer; nine examples
  (person-present, helmet, describe, phone-use, smoking, fall, fight, concealment,
  forklift-proximity). A folder of private checks overrides or adds to them.
- Keyframes from a clip: distinct frames around the event, whole frames, close-ups or both,
  with the geometry kept to map boxes between the source and each image.
- One backend for any OpenAI-compatible server (vLLM, Ollama, LM Studio, mlx-vlm, hosted),
  JSON Schema sent the way each expects, box format per model.
- Guardrails: parse, validate, one repair try, `no_verdict` instead of a guess; evidence must
  lie inside the detection in an image that was shown; review-only checks.
- CLI: `maia-vlm checks | ping | check`; `--fps` thins a clip like a gateway ring, `--frames`
  and `--max-side` shrink a check for one run.
- `maia-vlm run cases.yaml`: runs a list of cases, saves every verdict (JSONL) and a summary
  (accuracy, statuses, latency p50/p95, tokens); `sets/test-clips/cases.yaml`, 23 cases.
- `verdict_from`: a check derives its verdict from a specific answer field (helmet, phone-use,
  concealment); a self-contradicting answer is `inconsistent` and never becomes a confirmation.
- Close-ups stay centred on the subject at the frame's edge (padded, not slid back inside);
  evidence is held to the detector's box on the event frame only; prompts say which image
  that is.
- Evidence is one box, on the event image: grounding a box copied onto other images failed, and
  writing one per image cost seconds. `--save-images` writes what the model saw, boxes drawn.
- Run summaries count alerts (caught, missed, false alarms); `unclear` on a should-be-rejected
  case is `no alert`, not wrong. Timeouts are configurable (`--timeout`, default 120 s).
- Verdicts report `prompt_tokens`; a context-window error says how to raise the window.
