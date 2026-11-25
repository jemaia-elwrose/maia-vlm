# Changelog

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
