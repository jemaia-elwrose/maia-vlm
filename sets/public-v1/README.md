# public-v1: 46 cases, five checks, real events and deliberate negatives

Every check gets clips where the event happens (`expect: confirmed`) and clips where it
nearly does (`expect: rejected`): a model that says yes to everything scores badly here.

| check | real (pos) | not (neg) | from |
|---|---|---|---|
| fall | 5 falls | 5: sitting down, bending over, lying down on purpose | Le2i |
| fight | 5 fights | 5: hugs, sport, dancing, crowds | RLVS Violence / NonViolence |
| concealment | 5: an item into a bag or clothing | 5: into a basket, back on the shelf, kept in hand | Shoplifting / Normal |
| smoking | 5 smoking | 5: eating or drinking (a hand at the mouth) | HMDB51 smoke / eat, drink |
| helmet | 3 without a hard hat | 3 with one | PPE videos |

## Filling it (about an hour)

For each line of `cases.yaml`, pick a source video and cut the clip with the moment in the middle:

```bash
scripts/prepare-clip.sh <source video> <moment, seconds> sets/public-v1/media/<clip name>
scripts/prepare-clip.sh "Coffee_room/Videos/video (12).avi" 47.5 sets/public-v1/media/fall-pos-01.mp4
```

It cuts 10 s around the moment (H.264, 1280 px wide at most, 25 fps). A short source (HMDB51 and
RLVS clips are 2-6 s) is kept whole, and the script prints the `at:` to set if the moment is not
in the middle. Then note the original file name and start in `SOURCES.md`.

Picking well:

- **Negatives should be close calls**, not empty scenes: a person lying on a sofa for `fall`,
  two people hugging for `fight`, a hand at the mouth with a cup for `smoking`.
- **One person in focus** for `helmet`, `smoking` and `concealment`: the box comes from YOLO's
  largest person at the case time (`box: auto`; fights use `auto-group`, everyone), so a clip where that is someone else tests the wrong
  person. `--save-images` shows which person was picked (green box).
- **Le2i**: its annotation files give the fall's frame numbers; frame / 25 = seconds.
- A case you cannot fill yet: comment it out with `#`. Missing clips are never skipped
  silently: the run stops and names them.

## Running it

```bash
maia-vlm run sets/public-v1/cases.yaml --infer tcp://localhost:5555 --save-images
```

`--infer` is the maia-infer broker with a YOLO worker (as for the live test). Each result row
records the box YOLO gave and the time used, so a wrong box shows up in the results, not as
a mystery.

To run the same cases on another server with exactly the same input, reuse the times and boxes
of an earlier run: no detector needed.

```bash
maia-vlm run sets/public-v1/cases.yaml --boxes-from results/<earlier>.jsonl --url ... --model ...
scripts/mlx-run.sh results/<earlier>.jsonl        # Apple Silicon: starts mlx-vlm, runs, logs
```

Results so far: [RESULTS.md](RESULTS.md); the raw verdicts of each run are in `results/`.
