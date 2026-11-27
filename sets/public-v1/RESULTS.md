# public-v1 results

Raw verdicts: `results/` (one `.jsonl` row per case, with the model's full reply, and a `.md` summary).

46 cases, five checks (`fall` v4, `fight` v1, `concealment` v2, `smoking` v1, `helmet` v4), the
same model on two machines. The M4 run reused the T4 run's times and boxes (`--boxes-from`), so
the two differ only in the server, the chip and the weights' precision. Run on 2026-11-27.

| | Tesla T4 16 GB | Apple M4 MacBook Air 16 GB |
|---|---|---|
| server | vLLM | mlx-vlm 0.8.1 |
| model | Qwen3-VL-4B-Instruct, fp16 | Qwen3-VL-4B-Instruct, 8-bit |
| JSON schema | enforced by the server | in the prompt only |
| correct | **43 / 46 (93.5%)** | **41 / 46 (89.1%)** |
| real events caught | 22 / 23 | 22 / 23 |
| false alarms | 2 / 23 (8.7%) | 4 / 23 (17.4%) |
| latency p50 / p95 | 5.1 s / 8.0 s | 10.5 s / 13.8 s |
| whole set | 4.4 min | 8.1 min |
| prompt tokens (mean) | 1,366 | 1,366 |

Per check (correct out of cases):

| check | T4 | M4 |
|---|---|---|
| fall | 8 / 10 | 8 / 10 |
| fight | 10 / 10 | 10 / 10 |
| concealment | 10 / 10 | 9 / 10 |
| smoking | 10 / 10 | 9 / 10 |
| helmet | 5 / 6 | 5 / 6 |

## What went wrong, and why

On both machines:

- **`fall-pos-04`** (missed fall): no person detected at the moment, so the check ran on the
  whole scene; the model saw the person crouch onto the mat and answered `lowered_self`.
- **`fall-neg-01`** (false alarm): a person crouching on the floor reads as "fell to the floor".
  Getting down fast and getting down on purpose look alike in 8 frames at 448 px.
- **`ppe-pos-01`** (false alarm): a hard hat worn over a hood. The model reports the hood and
  "no hard hat" however the question is asked.

On the M4 only:

- **`theft-neg-03`**: "places it into a gray bag" where the person keeps the item in hand.
- **`smoke-neg-05`**: a hand near the mouth, "suggesting" a cigarette that the model never saw.

Every reply on the M4 was valid JSON even without an enforced schema. Prompt tokens are the
same on both servers (the same processor and image sizes); Ollama used about six times as
many for the same images, which is most of why it was slow.

## Swapping the detector (T4, maia-vlm 0.1.3)

The same set with `box: auto` boxes from maia-infer's RF-DETR (medium, Apache-2.0) instead
of YOLO: **43 / 46** again. RF-DETR found the person in `fall-pos-04`, which YOLO missed (that
case is now right), and `smoke-neg-05` turned wrong (a hand near the mouth read as smoking,
the M4 run's mistake). The detector changes which person and box the VLM sees, not how well
the checks do. RF-DETR medium ran at 37 ms per 640x360 frame (p50) on the T4.

## Tuning history (T4)

The checks were tuned on this set, so these numbers flatter them; they are a development
score, not a held-out one.

| run | change | correct |
|---|---|---|
| 1 | first run: `box: auto` = the most confident person; `fall` v2 | 37 / 46 |
| 2 | `box: auto` (largest person, ±0.8 s), `auto-group` for fights, `fall` v3 | 42 / 46 |
| 3 | `fall` v4 (`landed_on`), `helmet` v4 (`hard_hat` first) | 42 / 46 |
| 4 | `ppe-pos-03` box set by hand (auto picked a man facing away) | 43 / 46 |

The checks are frozen for 0.1.2. The remaining errors are the model's limits at this size;
0.2 fine-tunes for them.
