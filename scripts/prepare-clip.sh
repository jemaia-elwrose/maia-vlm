#!/usr/bin/env bash
# Cut a test clip so its moment sits in the middle: H.264, no audio, at most 1280 px wide, 25 fps.
#
#   scripts/prepare-clip.sh <source video> <moment in seconds> <output.mp4> [length, default 10]
#   scripts/prepare-clip.sh Coffee_room_01.avi 47.5 sets/public-v1/media/fall-pos-01.mp4
#
# The clip runs from moment - length/2 to moment + length/2, so the case's default 'at' (the
# middle) is the moment itself. A source shorter than that is kept whole (re-encoded).
set -euo pipefail
[ $# -ge 3 ] || { sed -n 2,9p "$0"; exit 1; }
SRC=$1 AT=$2 OUT=$3 LEN=${4:-10}
START=$(python3 -c "print(max(0.0, $AT - $LEN / 2))")
mkdir -p "$(dirname "$OUT")"
ffmpeg -hide_banner -loglevel error -y -ss "$START" -i "$SRC" -t "$LEN" -an \
    -vf "scale='min(1280,iw)':-2,fps=25" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -bf 0 "$OUT"
DUR=$(python3 -c "print(round(float('$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")'), 2))")
echo "$OUT: $DUR s from $START s of $(basename "$SRC") (middle = $(python3 -c "print(round($DUR / 2, 2))") s)"
[ "$(python3 -c "print(int(abs($DUR / 2 - ($AT - $START)) > 0.6))")" = 0 ] || \
    echo "  note: the source ends early, so the moment is not in the middle: set at: $(python3 -c "print(round($AT - $START, 2))") on this case"
