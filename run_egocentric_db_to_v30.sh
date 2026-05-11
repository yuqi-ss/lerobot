#!/usr/bin/env bash
# Convert egocentric_dataset_clips (Ego4D / Egocentric-100K) directly from
# Supabase into LeRobot v3.0 datasets.
#
# Single-process baseline.
# Layout:
#   * frames forced to 640x480
#   * state/action = 16D EEF [L_pos(3), L_quat(4), L_grip, R_pos(3), R_quat(4), R_grip]
#   * source artifacts staged to --work-dir on local NVMe (avoids ossfs FUSE
#     seek latency); each finished chunk cp'd to --output-root and rm'd.
#
# Resume: re-running with the same --output-root + --partition silently skips
# every record that belongs to an already-finalized chunk.

set -e

PY=/root/miniconda3/envs/ego/bin/python
export PYTHONPATH=/home/yuqi/lerobot/src
SCRIPT=/home/yuqi/lerobot/src/lerobot/scripts/egocentric_db_to_v30.py

OUT_ROOT=/home/ss-oss1/data/user/yuqi/data/egocentric_dataset
WORK_DIR=/home/yuqi/tmp/egocentric_v30_work
LOG_DIR=/home/yuqi/lerobot/logs/egocentric_v30
mkdir -p "$LOG_DIR" "$WORK_DIR"

PARTITION=Ego4D            # or Egocentric-100K
# This machine has 4 RTX 5880 Ada GPUs. Use one shard per GPU and NVENC for speed.
VCODEC=h264_nvenc
# Keep chunks modest because /home/yuqi/tmp currently has limited free space.
EPISODES_PER_CHUNK=1000
PAGE_SIZE=500
PARALLEL_SHARDS=4
GPU_COUNT=4
PREPROCESS_WORKERS=2
MAX_PREPARED_IN_FLIGHT=4
TOTAL_EPISODES=237555
PROGRESS_INTERVAL_SECONDS=30
# Use a wide stride so parallel shard chunk names never collide in OUT.
CHUNK_INDEX_STRIDE=${CHUNK_INDEX_STRIDE:-10000}

case "$PARTITION" in
  Ego4D) OUT="$OUT_ROOT/ego4d/" ;;
  Egocentric-100K) OUT="$OUT_ROOT/ego100k/" ;;
  *) OUT="$OUT_ROOT/${PARTITION,,}/" ;;
esac

echo "[info] partition=$PARTITION vcodec=$VCODEC"
echo "[info] work=$WORK_DIR  output=$OUT"

# ── debug: 5 episodes only ──
# "$PY" "$SCRIPT" \
#   --partition "$PARTITION" \
#   --output-root /tmp/ego4d_v30_test/ \
#   --work-dir   "$WORK_DIR" \
#   --episodes-per-chunk 5 --max-episodes 5 --overwrite \
#   --vcodec "$VCODEC"

# ── single host runner ──
# Uses 4 local shards by default; each shard is pinned to one GPU.
if [[ "$PARALLEL_SHARDS" -le 1 ]]; then
  "$PY" "$SCRIPT" \
    --partition "$PARTITION" \
    --output-root "$OUT" \
    --work-dir   "$WORK_DIR" \
    --episodes-per-chunk "$EPISODES_PER_CHUNK" \
    --page-size "$PAGE_SIZE" \
    --preprocess-workers "$PREPROCESS_WORKERS" \
    --max-prepared-in-flight "$MAX_PREPARED_IN_FLIGHT" \
    --vcodec "$VCODEC"
else
  pids=()
  start_ts=$(date +%s)
  for ((shard=0; shard<PARALLEL_SHARDS; shard++)); do
    shard_log="$LOG_DIR/${PARTITION}_shard${shard}_of_${PARALLEL_SHARDS}.log"
    echo "[info] starting shard=$shard/$PARALLEL_SHARDS log=$shard_log"
    CUDA_VISIBLE_DEVICES="$((shard % GPU_COUNT))" "$PY" "$SCRIPT" \
      --partition "$PARTITION" \
      --output-root "$OUT" \
      --work-dir   "$WORK_DIR" \
      --episodes-per-chunk "$EPISODES_PER_CHUNK" \
      --page-size "$PAGE_SIZE" \
      --preprocess-workers "$PREPROCESS_WORKERS" \
      --max-prepared-in-flight "$MAX_PREPARED_IN_FLIGHT" \
      --vcodec "$VCODEC" \
      --num-shards "$PARALLEL_SHARDS" \
      --shard "$shard" \
      --start-chunk-index "$((shard * CHUNK_INDEX_STRIDE))" \
      >"$shard_log" 2>&1 &
    pids+=("$!")
  done

  print_total_progress() {
    "$PY" - "$LOG_DIR" "$PARTITION" "$PARALLEL_SHARDS" "$TOTAL_EPISODES" "$start_ts" <<'PY'
import re
import sys
import time
from pathlib import Path

log_dir = Path(sys.argv[1])
partition = sys.argv[2]
num_shards = int(sys.argv[3])
total = int(sys.argv[4])
start_ts = int(sys.argv[5])

done_total = 0
failed_total = 0
skipped_total = 0
per_shard = []
for shard in range(num_shards):
    log_path = log_dir / f"{partition}_shard{shard}_of_{num_shards}.log"
    done = failed = skipped = 0
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        matches = list(re.finditer(r"done=(\d+), failed=(\d+), skipped=(\d+)", text))
        if matches:
            last = matches[-1]
            done, failed, skipped = map(int, last.groups())
    done_total += done
    failed_total += failed
    skipped_total += skipped
    per_shard.append(f"s{shard}:d{done}/f{failed}/sk{skipped}")

elapsed = max(1, int(time.time()) - start_ts)
rate = done_total / elapsed
remaining = max(0, total - done_total)
eta_seconds = int(remaining / rate) if rate > 0 else 0

def fmt_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

pct = 100.0 * done_total / total if total > 0 else 0.0
print(
    f"[total-progress] done={done_total}/{total} ({pct:.2f}%) "
    f"failed={failed_total} skipped={skipped_total} "
    f"speed={rate:.2f} ep/s elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta_seconds)} "
    f"| {' '.join(per_shard)}",
    flush=True,
)
PY
  }

  while true; do
    running=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        running=1
        break
      fi
    done
    print_total_progress
    if [[ "$running" -eq 0 ]]; then
      break
    fi
    sleep "$PROGRESS_INTERVAL_SECONDS"
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  exit "$status"
fi
