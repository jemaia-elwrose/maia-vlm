#!/usr/bin/env bash
# Run a case set against a local mlx-vlm server (Apple Silicon), on exactly the same
# times and boxes as an earlier run, so the two reports differ only in the server.
#
#   scripts/mlx-run.sh <earlier results .jsonl> [cases.yaml]
#
# Environment (all optional):
#   MODEL   mlx model to serve        (default mlx-community/Qwen3-VL-4B-Instruct-8bit)
#   PORT    server port               (default 8081)
#   TIMEOUT seconds per reply         (default 300)
#   SCHEMA  response_format | none    (default none: the schema is in the prompt, not enforced)
#
# Everything goes to logs/mlx-<time>/: server.log, run.log, system.txt, the results
# .jsonl/.md, and the images of each case.
set -euo pipefail

BOXES=${1:?usage: scripts/mlx-run.sh <earlier results .jsonl> [cases.yaml]}
CASES=${2:-sets/public-v1/cases.yaml}
MODEL=${MODEL:-mlx-community/Qwen3-VL-4B-Instruct-8bit}
PORT=${PORT:-8081}
TIMEOUT=${TIMEOUT:-300}
SCHEMA=${SCHEMA:-none}
LOG=logs/mlx-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/run.log"; }

[ -f "$BOXES" ] || { echo "no such file: $BOXES"; exit 1; }
[ -f "$CASES" ] || { echo "no such file: $CASES"; exit 1; }
python -c "import mlx_vlm" 2>/dev/null || { echo "mlx-vlm is not installed in this Python (pip install -U mlx-vlm)"; exit 1; }
command -v maia-vlm >/dev/null || { echo "maia-vlm is not installed in this Python (pip install -e .)"; exit 1; }
if curl -sf "http://localhost:$PORT/health" >/dev/null; then
  echo "something already answers on port $PORT; stop it or set PORT"; exit 1
fi

{
  echo "date:     $(date)"
  echo "machine:  $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
  echo "memory:   $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
  echo "macOS:    $(sw_vers -productVersion 2>/dev/null || uname -r)"
  echo "python:   $(python --version 2>&1)"
  echo "mlx:      $(python -c 'import mlx.core as m; print(m.__version__)' 2>/dev/null)"
  echo "mlx-vlm:  $(python -c 'import mlx_vlm; print(mlx_vlm.__version__)' 2>/dev/null)"
  echo "maia-vlm: $(python -c 'import importlib.metadata as m; print(m.version("maia-vlm"))')"
  echo "model:    $MODEL"
  echo "schema:   $SCHEMA"
  echo "boxes:    $BOXES"
  echo "cases:    $CASES"
} | tee "$LOG/system.txt"

say "starting mlx-vlm server on port $PORT (the first start downloads the model)"
python -m mlx_vlm.server --model "$MODEL" --port "$PORT" >"$LOG/server.log" 2>&1 &
SERVER=$!
stop_server() {
  pkill -TERM -P $SERVER 2>/dev/null || true
  kill -TERM $SERVER 2>/dev/null || true
  for _ in $(seq 1 10); do kill -0 $SERVER 2>/dev/null || return 0; sleep 1; done
  pkill -KILL -P $SERVER 2>/dev/null || true
  kill -KILL $SERVER 2>/dev/null || true
}
trap stop_server EXIT

for _ in $(seq 1 600); do
  curl -sf "http://localhost:$PORT/health" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { say "server exited; see $LOG/server.log"; tail -20 "$LOG/server.log"; exit 1; }
  sleep 2
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { say "server not up after 20 min"; exit 1; }

# /health answers while the model is still loading (chat then returns 503 model_loading):
# wait until a one-token text request succeeds
say "waiting for the model to load (the first start downloads it)"
PING="{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1}"
for _ in $(seq 1 600); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d "$PING" \
    "http://localhost:$PORT/v1/chat/completions" || echo 000)
  [ "$CODE" = 200 ] && break
  kill -0 $SERVER 2>/dev/null || { say "server exited; see $LOG/server.log"; tail -20 "$LOG/server.log"; exit 1; }
  [ "$CODE" = 503 ] || [ "$CODE" = 000 ] || { say "unexpected HTTP $CODE from the server"; tail -20 "$LOG/server.log"; exit 1; }
  sleep 2
done
[ "$CODE" = 200 ] || { say "model not loaded after 20 min"; exit 1; }
say "model loaded"

ARGS=(--url "http://localhost:$PORT/v1" --model "$MODEL" --schema-via "$SCHEMA" --timeout "$TIMEOUT")
say "warm-up: one case, not counted (loads the model and compiles kernels)"
FIRST=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['cases'][0]['id'])" "$CASES")
maia-vlm run "$CASES" --boxes-from "$BOXES" --only "$FIRST" --out "$LOG/warmup" "${ARGS[@]}" >>"$LOG/run.log" 2>&1 \
  || { say "warm-up failed; see $LOG/run.log"; tail -20 "$LOG/run.log"; exit 1; }
if grep -q '"status": "no_verdict"' "$LOG"/warmup/*.jsonl; then
  say "warm-up got no verdict from the server, so the run would be all no_verdict. Why:"
  python -c "import json, sys; print('   ', json.loads(open(sys.argv[1]).readline())['reason'])" "$(ls "$LOG"/warmup/*.jsonl)" | tee -a "$LOG/run.log"
  say "last lines of $LOG/server.log:"
  tail -25 "$LOG/server.log"
  exit 1
fi

say "running $CASES"
maia-vlm run "$CASES" --boxes-from "$BOXES" --out "$LOG" --save-images "${ARGS[@]}" 2>&1 | tee -a "$LOG/run.log"

RSS_KB=$(ps -o rss= -p $SERVER 2>/dev/null | tr -d ' ' || echo 0)
echo "server resident memory after the run: $(( ${RSS_KB:-0} / 1024 )) MB (approximate on unified memory)" | tee -a "$LOG/system.txt"
say "done: $LOG"
