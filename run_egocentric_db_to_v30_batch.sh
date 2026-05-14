#!/usr/bin/env bash
# Batch-convert egocentric_dataset_clips into many independent v3.0 datasets,
# one per path-range subset. Mirrors the structure of run_cobot_magic_raw_batch.sh
# (per-subset log, parallel scheduling, iowait/disk/net status, resume-skip).
#
# Output layout (no merge):
#   ${DST_ROOT}/
#     subset_0000_v30/  meta/  data/  videos/  annotations/
#     subset_0001_v30/  ...
#     ...
#
# Each subset_NNNN_v30/ is a self-contained LeRobot v3.0 dataset and can be
# loaded / trained on independently, exactly like the directories produced by
# run_cobot_magic_raw_batch.sh.
{

PREFIX=${1:-"subset"}
MAX_PARALLEL=${2:-64}

# ── user config ───────────────────────────────────────────────────────────────
# Defaults tuned for the host: 384 cores, 4×RTX 5880 Ada, ~42 GB local scratch.
# Single-subset peak ≈ (clips_per_subset × 3.3 MB).  With NUM_SUBSETS=256 each
# subset is ≈ 930 ep ≈ 3 GB; 12 parallel → ~36 GB local peak (safe under 42 GB).
PARTITION=${PARTITION:-Egocentric-100K}                # or Egocentric-100K
NUM_SUBSETS=${NUM_SUBSETS:-4096}              # how many path-range pieces to cut
VCODEC=${VCODEC:-h264_nvenc}
GPU_COUNT=${GPU_COUNT:-4}                    # NVENC sessions are unlimited on 5880 Ada
PREPROCESS_WORKERS=${PREPROCESS_WORKERS:-6}
MAX_PREPARED_IN_FLIGHT=${MAX_PREPARED_IN_FLIGHT:-12}
PAGE_SIZE=${PAGE_SIZE:-500}
OVERWRITE=${OVERWRITE:-false}
REDIRECT_LOG=${REDIRECT_LOG:-true}
# Multi-machine sharding: each shard processes subsets where i % NUM_SHARDS == SHARD.
# All shards must use the same NUM_SUBSETS / PARTITION / DST_ROOT (so that subset_NNNN
# names line up across machines and outputs naturally merge into one DST_ROOT).
SHARD=${SHARD:-6}
NUM_SHARDS=${NUM_SHARDS:-8}
# ─────────────────────────────────────────────────────────────────────────────

if (( SHARD < 0 || SHARD >= NUM_SHARDS )); then
    echo "Invalid SHARD=${SHARD} (must be in [0, ${NUM_SHARDS}))" >&2
    exit 2
fi

PY=/root/miniconda3/envs/ego/bin/python
[[ -x "${PY}" ]] || PY=python
export PYTHONPATH=/home/yuqi/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}

SCRIPT=/home/yuqi/lerobot/src/lerobot/scripts/egocentric_db_subset_to_v30.py

DST_ROOT=${DST_ROOT:-/home/ss-oss1/data/user/yuqi/data/egocentric_dataset_subsets/${PARTITION,,}_v30}
WORK_ROOT=${WORK_ROOT:-/home/yuqi/tmp/egocentric_v30_work}
LOG_DIR=${LOG_DIR:-/home/yuqi/lerobot/logs/egocentric_v30_batch}

mkdir -p "${DST_ROOT}" "${WORK_ROOT}" "${LOG_DIR}"

NET_DEV=eth0
DISK_DEV=nvme0n1

# ── helpers ──────────────────────────────────────────────────────────────────
_CACHE_TIME=0
_C_IOWAIT=0
_C_DISK=0
_C_RX=0
_C_TX=0

_refresh_stats() {
    local now; now=$(date +%s)
    (( now - _CACHE_TIME >= 2 )) || return
    local a b rx0 tx0 rx1 tx1 ticks_a ticks_b
    read -ra a < /proc/stat
    ticks_a=$(awk "/${DISK_DEV} /{print \$13}" /proc/diskstats 2>/dev/null)
    read rx0 tx0 < <(awk -v d="${NET_DEV}" '$0 ~ d":" {print $2, $10}' /proc/net/dev 2>/dev/null)
    sleep 0.3
    read -ra b < /proc/stat
    ticks_b=$(awk "/${DISK_DEV} /{print \$13}" /proc/diskstats 2>/dev/null)
    read rx1 tx1 < <(awk -v d="${NET_DEV}" '$0 ~ d":" {print $2, $10}' /proc/net/dev 2>/dev/null)
    local total_a=0 total_b=0
    for v in "${a[@]:1}"; do (( total_a += v )); done
    for v in "${b[@]:1}"; do (( total_b += v )); done
    local td=$(( total_b - total_a ))
    _C_IOWAIT=$(( (${b[5]:-0} - ${a[5]:-0}) * 100 / (td + 1) ))
    local dt=$(( ${ticks_b:-0} - ${ticks_a:-0} ))
    _C_DISK=$(( dt * 100 / 300 )); (( _C_DISK > 100 )) && _C_DISK=100
    _C_RX=$(( ((${rx1:-0}) - (${rx0:-0})) * 10 / 3 / 1048576 ))
    _C_TX=$(( ((${tx1:-0}) - (${tx0:-0})) * 10 / 3 / 1048576 ))
    _CACHE_TIME=$now
}

print_status() {
    _refresh_stats
    local running=${#PIDS[@]}
    local free_gb
    free_gb=$(df -BG "${WORK_ROOT}" | awk 'NR==2 {gsub("G","",$4); print $4}')
    echo ""
    echo "── ${DONE_COUNT}/${MY_SHARD_TOTAL:-$TOTAL} done [shard ${SHARD}/${NUM_SHARDS}] | ${running} running | iowait ${_C_IOWAIT}% | disk ${_C_DISK}% | net ↓${_C_RX} ↑${_C_TX} MB/s | scratch_free ${free_gb}G ──"
    local i prog
    for i in "${!PIDS[@]}"; do
        prog=""
        if [[ -f "${LOGS[$i]}" ]]; then
            # Worker logs `[subset_xxxx] done=N failed=M elapsed=Ts` every ~60s.
            prog=$(grep -oE 'done=[0-9]+ failed=[0-9]+ elapsed=[0-9.]+s' "${LOGS[$i]}" | tail -n1)
        fi
        printf "  [running]  %-16s  %s\n" "${NAMES[$i]}" "${prog}"
    done
}

# ── enumerate path boundaries ────────────────────────────────────────────────
echo "Sampling path boundaries for partition=${PARTITION}, subsets=${NUM_SUBSETS} ..."

BOUNDARY_FILE=$(mktemp)
"${PY}" - "${PARTITION}" "${NUM_SUBSETS}" >"${BOUNDARY_FILE}" <<'PYEOF'
import sys
sys.path.insert(0, "/home/yuqi/lerobot/src")
from lerobot.scripts.egocentric_database_operation import get_egocentric_boundaries

partition = sys.argv[1]
n = int(sys.argv[2])
sample_interval = max(500, 240_000 // max(1, n * 4))
boundaries = get_egocentric_boundaries(
    partition=partition,
    total_machines=n,
    sample_interval=sample_interval,
    verbose=False,
)
if len(boundaries) < 2:
    raise SystemExit(f"got only {len(boundaries)} boundaries; partition empty?")
for b in boundaries:
    print(b)
PYEOF
ec=$?
if [[ $ec -ne 0 ]]; then
    echo "Failed to compute path boundaries (exit=$ec)" >&2
    rm -f "${BOUNDARY_FILE}"
    exit 1
fi

mapfile -t BOUNDARIES < "${BOUNDARY_FILE}"
rm -f "${BOUNDARY_FILE}"
TOTAL=$(( ${#BOUNDARIES[@]} - 1 ))

if (( TOTAL <= 0 )); then
    echo "No usable boundaries (got ${#BOUNDARIES[@]} entries)" >&2
    exit 1
fi

MY_SHARD_TOTAL=$(( (TOTAL + NUM_SHARDS - 1 - SHARD) / NUM_SHARDS ))
echo "Got ${#BOUNDARIES[@]} boundaries → ${TOTAL} subsets total, this shard=${SHARD}/${NUM_SHARDS} owns ${MY_SHARD_TOTAL} subsets, max ${MAX_PARALLEL} parallel, vcodec=${VCODEC}"
echo "Output dir: ${DST_ROOT}"
echo "Work dir  : ${WORK_ROOT}"
echo "Log dir   : ${LOG_DIR}"
echo ""

# ── main scheduler ───────────────────────────────────────────────────────────
PIDS=()
NAMES=()
LOGS=()
FAILED=()
SKIP_COUNT=0
DONE_COUNT=0
LAUNCH_COUNT=0
LAST_STATUS=$(date +%s)

_cleanup() {
    echo ""
    echo "Interrupted — killing ${#PIDS[@]} running job(s)..."
    [[ ${#PIDS[@]} -gt 0 ]] && kill "${PIDS[@]}" 2>/dev/null
    wait 2>/dev/null
    exit 130
}
trap '_cleanup' INT TERM

_reap_finished() {
    local i pid
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        if ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}"
            if [ $? -eq 0 ]; then
                echo "[DONE ]  ${NAMES[$i]}"
            else
                echo "[FAIL ]  ${NAMES[$i]}  (log: ${LOGS[$i]})"
                FAILED+=("${NAMES[$i]}")
            fi
            unset 'PIDS[$i]'; unset 'NAMES[$i]'; unset 'LOGS[$i]'
            DONE_COUNT=$(( DONE_COUNT + 1 ))
        fi
    done
    PIDS=( "${PIDS[@]}" ); NAMES=( "${NAMES[@]}" ); LOGS=( "${LOGS[@]}" )
}

_maybe_print_status() {
    local now; now=$(date +%s)
    if (( now - LAST_STATUS >= 10 )); then
        print_status
        LAST_STATUS=${now}
    fi
}

for ((i=0; i<TOTAL; i++)); do
    # Multi-machine sharding: skip subsets owned by other shards.
    (( i % NUM_SHARDS != SHARD )) && continue
    SUBSET_NAME=$(printf "%s_%04d" "${PREFIX}" "$i")
    OUT_DIR="${DST_ROOT}/${SUBSET_NAME}_v30"

    if [[ "${OVERWRITE}" != "true" && -f "${OUT_DIR}/meta/info.json" && -d "${OUT_DIR}/meta/episodes" ]]; then
        echo "[SKIP ]  ${SUBSET_NAME}  (already converted)"
        SKIP_COUNT=$(( SKIP_COUNT + 1 ))
        DONE_COUNT=$(( DONE_COUNT + 1 ))
        continue
    fi

    while true; do
        _reap_finished
        _maybe_print_status
        (( ${#PIDS[@]} < MAX_PARALLEL )) && break
        sleep 2
    done

    PATH_GTE="${BOUNDARIES[$i]}"
    PATH_LT="${BOUNDARIES[$i+1]}"
    GPU=$(( LAUNCH_COUNT % GPU_COUNT ))
    LAUNCH_COUNT=$(( LAUNCH_COUNT + 1 ))
    LOG="${LOG_DIR}/${SUBSET_NAME}.log"

    OVERWRITE_FLAG=(); [[ "${OVERWRITE}" == "true" ]] && OVERWRITE_FLAG=(--overwrite)

    echo "[START]  ${SUBSET_NAME}  gpu=${GPU}  range=[${PATH_GTE} , ${PATH_LT}]  (${DONE_COUNT}/${MY_SHARD_TOTAL} done in shard ${SHARD}/${NUM_SHARDS}, ${#PIDS[@]} running)"

    if [[ "${REDIRECT_LOG}" == "true" ]]; then
        CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${SCRIPT}" \
          --partition "${PARTITION}" \
          --subset-name "${SUBSET_NAME}" \
          --path-gte "${PATH_GTE}" \
          --path-lt "${PATH_LT}" \
          --output-root "${OUT_DIR}" \
          --work-dir "${WORK_ROOT}" \
          --page-size "${PAGE_SIZE}" \
          --preprocess-workers "${PREPROCESS_WORKERS}" \
          --max-prepared-in-flight "${MAX_PREPARED_IN_FLIGHT}" \
          --vcodec "${VCODEC}" \
          "${OVERWRITE_FLAG[@]}" \
          > "${LOG}" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${SCRIPT}" \
          --partition "${PARTITION}" \
          --subset-name "${SUBSET_NAME}" \
          --path-gte "${PATH_GTE}" \
          --path-lt "${PATH_LT}" \
          --output-root "${OUT_DIR}" \
          --work-dir "${WORK_ROOT}" \
          --page-size "${PAGE_SIZE}" \
          --preprocess-workers "${PREPROCESS_WORKERS}" \
          --max-prepared-in-flight "${MAX_PREPARED_IN_FLIGHT}" \
          --vcodec "${VCODEC}" \
          "${OVERWRITE_FLAG[@]}" &
    fi

    PIDS+=($!)
    NAMES+=("${SUBSET_NAME}")
    LOGS+=("${LOG}")
done

while [ "${#PIDS[@]}" -gt 0 ]; do
    _reap_finished
    _maybe_print_status
    sleep 2
done

echo ""
echo "========================================"
echo "  SHARD ${SHARD}/${NUM_SHARDS}  |  OWNED: ${MY_SHARD_TOTAL}  |  SKIP: ${SKIP_COUNT}  |  FAILED: ${#FAILED[@]}"
echo "========================================"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed subsets:"
    for D in "${FAILED[@]}"; do
        echo "  - ${D}  (log: ${LOG_DIR}/${D}.log)"
    done
    exit 1
fi

exit 0
}
