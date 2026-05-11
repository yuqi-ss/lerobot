#!/usr/bin/env bash
# Batch convert cobot_magic_raw datasets (v2.1 -> v3.0).
# Wrapped in { } so bash reads the entire file before executing,
# preventing corruption if the file is edited while running.
{

PREFIX=${1:-"chunk-"}
MAX_PARALLEL=${2:-"auto"}

# ── user config ───────────────────────────────────────────────────────────────
OVERWRITE=true       # true = pass --overwrite-output; false = skip already-converted datasets
REDIRECT_LOG=true    # true = write each job's output to LOG_DIR; false = print to terminal
# ─────────────────────────────────────────────────────────────────────────────

SRC_ROOT=/home/ss-oss1/data/user/mjc/data/egocentric_clips_lingbot_action30_short_caption/ego100k/
DST_ROOT=/home/ss-oss1/data/user/yuqi/data/egocentric_clips_lingbot_action30_short_caption/ego100k_v30/
PYTHON=/home/yuqi/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py
LOG_DIR=/home/yuqi/lerobot/logs && mkdir -p "${LOG_DIR}"

AUTO_HARD_CAP=8
NET_DEV=eth0         # network interface for ossfs traffic
DISK_DEV=nvme0n1     # local NVMe device

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
    ticks_a=$(awk "/${DISK_DEV} /{print \$13}" /proc/diskstats)
    read rx0 tx0 < <(awk -v d="${NET_DEV}" '$0 ~ d":" {print $2, $10}' /proc/net/dev)
    sleep 0.3
    read -ra b < /proc/stat
    ticks_b=$(awk "/${DISK_DEV} /{print \$13}" /proc/diskstats)
    read rx1 tx1 < <(awk -v d="${NET_DEV}" '$0 ~ d":" {print $2, $10}' /proc/net/dev)

    # iowait
    local total_a=0 total_b=0
    for v in "${a[@]:1}"; do (( total_a += v )); done
    for v in "${b[@]:1}"; do (( total_b += v )); done
    local td=$(( total_b - total_a ))
    _C_IOWAIT=$(( (${b[5]} - ${a[5]}) * 100 / (td + 1) ))

    # NVMe disk %util (ticks in ms, sample 300ms)
    local dt=$(( ticks_b - ticks_a ))
    _C_DISK=$(( dt * 100 / 300 )); (( _C_DISK > 100 )) && _C_DISK=100

    # network MB/s (bytes in 300ms → *10/3 for per-second)
    _C_RX=$(( (rx1 - rx0) * 10 / 3 / 1048576 ))
    _C_TX=$(( (tx1 - tx0) * 10 / 3 / 1048576 ))

    _CACHE_TIME=$now
}

effective_max() {
    if [[ "${MAX_PARALLEL}" == "auto" ]]; then
        echo "${AUTO_HARD_CAP}"
    else
        echo "${MAX_PARALLEL}"
    fi
}

print_status() {
    _refresh_stats
    local running=${#PIDS[@]}
    echo ""
    echo "── ${DONE_COUNT}/${TOTAL} done | ${running} running | iowait ${_C_IOWAIT}% | disk ${_C_DISK}% | net ↓${_C_RX} ↑${_C_TX} MB/s ──"
    for i in "${!PIDS[@]}"; do
        printf "  [running]  %s\n" "${NAMES[$i]}"
    done
}

# ── main ─────────────────────────────────────────────────────────────────────

_cleanup() {
    echo ""
    echo "Interrupted — killing ${#PIDS[@]} running job(s)..."
    [[ ${#PIDS[@]} -gt 0 ]] && kill "${PIDS[@]}" 2>/dev/null
    wait 2>/dev/null
    exit 130
}
trap '_cleanup' INT TERM

DATASETS=()
while IFS= read -r -d '' d; do
    DATASETS+=("$(basename "$d")")
done < <(find "${SRC_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${PREFIX}*" -print0 | sort -z)
TOTAL=${#DATASETS[@]}

if [ "${TOTAL}" -eq 0 ]; then
    echo "No datasets found matching prefix '${PREFIX}' under ${SRC_ROOT}"
    exit 1
fi

if [[ "${MAX_PARALLEL}" == "auto" ]]; then
    echo "Found ${TOTAL} datasets — auto parallel (cap=${AUTO_HARD_CAP})"
else
    echo "Found ${TOTAL} datasets — ${MAX_PARALLEL} parallel"
fi
echo ""

PIDS=()
NAMES=()
LOGS=()
FAILED=()
DONE_COUNT=0
LAST_STATUS=$(date +%s)

_reap_finished() {
    local i pid
    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        if ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}"
            if [ $? -eq 0 ]; then
                echo "[DONE ]  ${NAMES[$i]}"
            else
                echo "[FAIL ]  ${NAMES[$i]}"
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

SKIP_COUNT=0

for DATASET in "${DATASETS[@]}"; do
    OUT_DIR="${DST_ROOT}/${DATASET}_v30"

    # Skip completed datasets when not overwriting
    if [[ "${OVERWRITE}" != "true" && -f "${OUT_DIR}/meta/info.json" && ! -d "${OUT_DIR}/_parallel_tmp" ]]; then
        echo "[SKIP ]  ${DATASET}  (already converted)"
        SKIP_COUNT=$(( SKIP_COUNT + 1 ))
        DONE_COUNT=$(( DONE_COUNT + 1 ))
        continue
    fi

    while true; do
        _reap_finished
        _maybe_print_status
        slot_max=$(effective_max)
        (( ${#PIDS[@]} < slot_max )) && break
        sleep 2
    done

    LOG="${LOG_DIR}/${DATASET}.log"
    echo "[START]  ${DATASET}  (${DONE_COUNT}/${TOTAL} done, ${#PIDS[@]} running)"

    OVERWRITE_FLAG=(); [[ "${OVERWRITE}" == "true" ]] && OVERWRITE_FLAG=(--overwrite-output)

    if [[ "${REDIRECT_LOG}" == "true" ]]; then
        python "${PYTHON}" \
          --repo-id="${DATASET}" \
          --root="${SRC_ROOT}/${DATASET}/" \
          --output-root="${DST_ROOT}/${DATASET}_v30/" \
          --num-workers 0 \
          "${OVERWRITE_FLAG[@]}" \
          > "${LOG}" 2>&1 &
    else
        python "${PYTHON}" \
          --repo-id="${DATASET}" \
          --root="${SRC_ROOT}/${DATASET}/" \
          --output-root="${DST_ROOT}/${DATASET}_v30/" \
          --num-workers 0 \
          "${OVERWRITE_FLAG[@]}" &
    fi

    PIDS+=($!)
    NAMES+=("${DATASET}")
    LOGS+=("${LOG}")
done

# Drain remaining jobs
while [ "${#PIDS[@]}" -gt 0 ]; do
    _reap_finished
    _maybe_print_status
    sleep 2
done

echo ""
echo "========================================"
echo "  TOTAL: ${TOTAL}  |  SKIP: ${SKIP_COUNT}  |  FAILED: ${#FAILED[@]}"
echo "========================================"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed datasets:"
    for D in "${FAILED[@]}"; do
        hint_log=""
        [[ "${REDIRECT_LOG}" == "true" ]] && hint_log="  (log: ${LOG_DIR}/${D}.log)"
        echo "  - ${D}${hint_log}"
    done
    exit 1
fi

exit
}
