#!/usr/bin/env bash
# Live test of the whole chain, with every log kept:
#   camera (maia-gateway) -> detector (maia-infer RF-DETR, by reference) -> event -> maia-vlm -> verdict on the clip
#
# What must already run:  maia-gateway (Docker or `maia-gateway serve`) with the camera playing,
#                         and the VLM server (vLLM / Ollama) that MAIA_VLM_URL points at.
# What this script starts: maia-infer broker + $SERVICE worker (unless one answers already),
#                          maia-vlm serve, and the detector (maia-gateway's examples/detect_from_bus.py).
# One Python environment needs maia-vlm[live], maia-infer[rfdetr] and maia-gateway (see the README).
#
#   scripts/live-test.sh                        # camera demo-video, 60 s of detection
#   CAMERA=gate DURATION=120 scripts/live-test.sh
#
# Settings (environment): CAMERA, DURATION (s), COOLDOWN (s between events), RULES (a rules file;
# default: person -> person-present + describe), GATEWAY_DIR (the maia-gateway repo, for the
# detector), API, REDIS, INFER, SERVICE (rfdetr | yolo), MAIA_VLM_URL, MAIA_VLM_MODEL, FPS, SAVE_IMAGES=1, LOG_ROOT.
set -uo pipefail

CAMERA=${CAMERA:-demo-video}
DURATION=${DURATION:-60}
COOLDOWN=${COOLDOWN:-10}
API=${API:-http://localhost:8080}
REDIS=${REDIS:-redis://localhost:6379/0}
INFER=${INFER:-tcp://localhost:5555}
SERVICE=${SERVICE:-rfdetr}   # maia-infer detection worker: rfdetr (Apache-2.0) or yolo (AGPL-3.0)
GATEWAY_DIR=${GATEWAY_DIR:-../maia-gateway}
FPS=${FPS:-5}
SAVE_IMAGES=${SAVE_IMAGES:-1}
LOG_ROOT=${LOG_ROOT:-logs}
export MAIA_VLM_URL=${MAIA_VLM_URL:-http://localhost:8000/v1}
export MAIA_VLM_MODEL=${MAIA_VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}

# `timeout` is GNU coreutils: on macOS it is `gtimeout` (brew install coreutils)
TIMEOUT=$(command -v timeout || command -v gtimeout || true)
[ -n "$TIMEOUT" ] || { echo "needs 'timeout' (Linux) or 'gtimeout' (macOS: brew install coreutils)"; exit 1; }

RUN="$LOG_ROOT/live-$(date +%Y%m%d-%H%M%S)-$CAMERA"
mkdir -p "$RUN"
PIDS=()
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/run.log"; }
fail() { say "STOP: $*"; exit 1; }
cleanup() {
    for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT

say "run folder: $RUN"
say "camera $CAMERA, $DURATION s of detection, model $MAIA_VLM_MODEL at $MAIA_VLM_URL"

# ---------------------------------------------------------------- 1. preflight
python - <<'EOF' >>"$RUN/run.log" 2>&1 || fail "missing packages, see $RUN/run.log"
import importlib, importlib.util, sys
missing = [m for m in ("maia_vlm.live", "maia_infer", "maia_gateway.bus", "maia_ring", "redis") if not importlib.util.find_spec(m.split(".")[0])]
for m in ("maia_vlm.live", "maia_gateway.bus"):
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m} ({exc})")
print("python:", sys.executable)
if missing:
    print("missing:", ", ".join(missing))
    print('install: pip install -e "../maia-vlm[live]" -e "../maia-gateway" "maia-infer[rfdetr] @ git+https://github.com/jemaia-elwrose/maia-infer@v0.5.0"')
    sys.exit(1)
EOF
[ -f "$GATEWAY_DIR/examples/detect_from_bus.py" ] || fail "no detector at $GATEWAY_DIR/examples/detect_from_bus.py (set GATEWAY_DIR)"
curl -sf "$API/health" >"$RUN/gateway-health.json" || fail "maia-gateway does not answer at $API"
STATE=$(curl -sf "$API/cameras/$CAMERA" | python -c "import sys,json; print(json.load(sys.stdin)['state'])" 2>/dev/null)
[ "$STATE" = "running" ] || fail "camera $CAMERA is '${STATE:-missing}', not running (curl $API/cameras)"
maia-vlm ping >"$RUN/vlm-ping.txt" 2>&1 || fail "the VLM server does not answer at $MAIA_VLM_URL ($RUN/vlm-ping.txt)"
grep -q "<- configured" "$RUN/vlm-ping.txt" || say "warning: $MAIA_VLM_MODEL is not in the server's model list ($RUN/vlm-ping.txt)"
say "preflight ok: gateway, camera running, VLM server up"

# ---------------------------------------------------------------- 2. maia-infer (YOLO)
if maia-infer ping --endpoint "$INFER" >/dev/null 2>&1 && maia-infer stats --endpoint "$INFER" 2>/dev/null | grep -q "\"$SERVICE\""; then
    say "maia-infer already running with a $SERVICE worker: using it"
else
    say "starting maia-infer broker + $SERVICE worker (logs: infer-broker.log, infer-worker.log)"
    maia-infer broker >"$RUN/infer-broker.log" 2>&1 & PIDS+=($!)
    sleep 2
    maia-infer worker "$SERVICE" >"$RUN/infer-worker.log" 2>&1 & PIDS+=($!)
    for _ in $(seq 1 90); do
        maia-infer stats --endpoint "$INFER" 2>/dev/null | grep -q "\"$SERVICE\"" && break
        sleep 1
    done
    maia-infer stats --endpoint "$INFER" 2>/dev/null | grep -q "\"$SERVICE\"" || fail "the $SERVICE worker did not register (infer-worker.log)"
    say "$SERVICE worker ready"
fi

# ---------------------------------------------------------------- 3. maia-vlm serve
if [ -z "${RULES:-}" ]; then
    RULES="$RUN/rules.yaml"
    printf 'rules:\n  - {camera: "%s", label: person, checks: [person-present, describe]}\n' "$CAMERA" >"$RULES"
fi
cp "$RULES" "$RUN/rules.yaml" 2>/dev/null || true
IMG_ARGS=()
[ "$SAVE_IMAGES" = "1" ] && IMG_ARGS=(--save-images "$RUN/images")
say "starting maia-vlm serve (log: vlm.log, verdicts: verdicts.jsonl)"
maia-vlm serve --rules "$RULES" --redis "$REDIS" --gateway "$API" --fps "$FPS" \
    --log "$RUN/verdicts.jsonl" "${IMG_ARGS[@]}" >"$RUN/vlm.log" 2>&1 & VLM_PID=$!
PIDS+=($VLM_PID)
sleep 3
kill -0 $VLM_PID 2>/dev/null || fail "maia-vlm serve stopped at start (vlm.log)"

# ---------------------------------------------------------------- 4. detection
say "detecting for $DURATION s (log: detect.log); an event at most every $COOLDOWN s"
"$TIMEOUT" --signal=INT "$DURATION" python "$GATEWAY_DIR/examples/detect_from_bus.py" --camera "$CAMERA" \
    --label person --api "$API" --redis "$REDIS" --infer "$INFER" --service "$SERVICE" --cooldown "$COOLDOWN" >"$RUN/detect.log" 2>&1
EVENTS=$(grep -c " -> clip " "$RUN/detect.log" || true)
say "detection done: $EVENTS event(s)"

# ---------------------------------------------------------------- 5. let the last checks finish
WAIT=$((COOLDOWN + 15))
say "waiting up to $WAIT s for the last verdicts"
EXPECTED=$((EVENTS * $(python -c "import yaml,sys; r=yaml.safe_load(open('$RULES'))['rules']; print(sum(len(x['checks']) for x in r if x.get('label') in ('person','*')))")))
for _ in $(seq 1 "$WAIT"); do
    DONE=$(wc -l <"$RUN/verdicts.jsonl" 2>/dev/null || echo 0)
    [ "$DONE" -ge "$EXPECTED" ] && break
    sleep 1
done
kill -TERM $VLM_PID 2>/dev/null; wait $VLM_PID 2>/dev/null

# ---------------------------------------------------------------- 6. report
# a clip is listed once it is written, post_s (default 10 s) after its last event
say "waiting up to 40 s for the clips to finish recording"
for _ in $(seq 1 40); do
    curl -sf "$API/clips?camera=$CAMERA&limit=100" >"$RUN/clips.json" || echo "[]" >"$RUN/clips.json"
    python - "$RUN" <<'EOF2' && break
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
want = {json.loads(l)["clip_id"] for l in (run / "verdicts.jsonl").read_text().splitlines()} if (run / "verdicts.jsonl").exists() else set()
have = {c["id"] for c in json.loads((run / "clips.json").read_text())}
sys.exit(0 if want <= have else 1)
EOF2
    sleep 1
done
python - "$RUN" "$EVENTS" <<'EOF' | tee "$RUN/summary.txt"
import json, statistics, sys
from pathlib import Path
run, events = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in (run / "verdicts.jsonl").read_text().splitlines()] if (run / "verdicts.jsonl").exists() else []
print(f"\n=== live test: {run.name}")
print(f"events raised by the detector: {events}; verdicts: {len(rows)}")
by = {}
for r in rows:
    by.setdefault(r["check"], []).append(r)
for check, rs in sorted(by.items()):
    lat = [r["latency_ms"] for r in rs if r["latency_ms"]]
    verdicts = {}
    for r in rs:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    src = {}
    for r in rs:
        src[r["frames_from"]] = src.get(r["frames_from"], 0) + 1
    p50 = f"{statistics.median(lat) / 1000:.1f} s" if lat else "-"
    print(f"  {check:<16} {len(rs):>3} checks  verdicts {verdicts}  frames {src}  latency p50 {p50}")
odd = [r for r in rows if r["status"] != "ok"]
for r in odd:
    print(f"  ! {r['clip_id']} {r['check']}: {r['verdict']} ({r['status']}) {r['reason'][:120]}")
clips = json.loads((run / "clips.json").read_text())
mine = {r["clip_id"] for r in rows}
with_v = [c for c in clips if c["id"] in mine and c.get("verdicts")]
print(f"clips from this run: {len(mine)} (events close together share a clip); with verdicts attached: {len(with_v)}")
for c in with_v[:5]:
    for v in c["verdicts"]:
        print(f"  {c['id']}  {v['check']}: {v['verdict']} - {v.get('description', '')[:100]}")
ok = events > 0 and len(rows) > 0 and not [r for r in rows if r["status"] == "no_verdict"]
print("RESULT:", "PASS" if ok else "CHECK THE LOGS", f"(logs and images in {run})")
EOF
