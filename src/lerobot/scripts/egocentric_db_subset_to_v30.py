"""Convert one path-range subset of egocentric_dataset_clips to a single v3.0 dataset.

This is the building block for the parallel batch runner
(`run_egocentric_db_to_v30_batch.sh`): each invocation processes a half-open
path range `[--path-gte, --path-lt)` and produces ONE self-contained LeRobot
v3.0 dataset under `--output-root`. Different invocations write to different
output directories and never share state, so they can run in parallel without
collision (analogous to how `run_cobot_magic_raw_batch.sh` produces one v3.0
dataset per source dataset).

Internal v3.0 chunking (`data/chunk-NNN/file-NNN.parquet`,
`videos/<key>/chunk-NNN/file-NNN.mp4`, etc.) is delegated entirely to
`LeRobotDataset` so the on-disk layout matches what `convert_dataset_v21_to_v30`
emits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Reuse all heavy-lifting helpers from the existing module so we have a single
# source of truth for the per-clip pipeline (asset staging, IK resampling,
# pose3d slimming, annotation building, dataset opening, frame writing).
from lerobot.scripts.egocentric_db_to_v30 import (
    PARTITION_EGO100K,
    PARTITION_EGO4D,
    _iter_prepared_episodes_in_order,
    _iter_records,
    _open_chunk_dataset,
    _write_episode_to_dataset,
    build_annotation,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", required=True, choices=[PARTITION_EGO4D, PARTITION_EGO100K])
    p.add_argument("--subset-name", required=True,
                   help="Used as the output dataset id and basename of work dir.")
    p.add_argument("--path-gte", type=str, default=None)
    p.add_argument("--path-lt", type=str, default=None)
    p.add_argument("--output-root", required=True, type=Path,
                   help="Final v3.0 dataset directory, e.g. .../<subset>_v30/.")
    p.add_argument("--work-dir", required=True, type=Path,
                   help="Local scratch root (real POSIX FS, not ossfs).")
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--preprocess-workers", type=int, default=2)
    p.add_argument("--max-prepared-in-flight", type=int, default=4)
    p.add_argument("--vcodec", default="h264_nvenc")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Debug cap on the number of episodes for this subset.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _is_dataset_complete(output_root: Path) -> bool:
    return (
        (output_root / "meta" / "info.json").exists()
        and (output_root / "meta" / "episodes").is_dir()
    )


def _silence_third_party_loggers() -> None:
    for noisy in ("httpx", "httpcore", "hpack", "urllib3", "postgrest"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        import av

        av.logging.set_level(av.logging.PANIC)
        av.logging.set_libav_level(av.logging.PANIC)
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("SVT_LOG", "1")
    os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
    try:
        import datasets as _hfds

        _hfds.disable_progress_bar()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _silence_third_party_loggers()

    output_root: Path = args.output_root
    if _is_dataset_complete(output_root) and not args.overwrite:
        logger.info("subset %s already complete at %s, skipping",
                    args.subset_name, output_root)
        return 0

    if output_root.exists():
        if not args.overwrite:
            # Partially-written remote dir without info.json: clean and retry.
            logger.warning("output %s exists but is incomplete, wiping for retry",
                           output_root)
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    # Per-subset local scratch; built fresh each invocation. ffmpeg streaming
    # encoder needs a real POSIX FS so we always write here first, then cp -r
    # to output_root at the end (ossfs-safe move).
    work_root: Path = args.work_dir / f"{args.subset_name}_work"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    dataset_root = work_root / args.subset_name
    src_cache_root = work_root / "_src_cache"
    src_cache_root.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(args.work_dir).free / (1 << 30)
    logger.info(
        "subset=%s partition=%s path_range=[%s, %s) "
        "output=%s work=%s free=%.1fGB workers=%d in_flight=%d vcodec=%s",
        args.subset_name, args.partition, args.path_gte, args.path_lt,
        output_root, work_root, free_gb,
        args.preprocess_workers, args.max_prepared_in_flight, args.vcodec,
    )

    dataset = _open_chunk_dataset(
        repo_id=args.subset_name,
        root=dataset_root,
        vcodec=args.vcodec,
    )
    annotations_dir = dataset_root / "annotations" / "chunk-000"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    record_seq = 0

    def _record_iter():
        nonlocal record_seq
        for record in _iter_records(
            args.partition, args.path_gte, args.path_lt,
            args.page_size, args.max_episodes,
        ):
            cache_dir = src_cache_root / (
                f"rec_{record_seq:08d}_{str(record.get('id', ''))[:12]}"
            )
            record_seq += 1
            yield record, cache_dir

    total_done = 0
    total_failed = 0
    t0 = time.time()
    last_log = t0

    try:
        for prepared in _iter_prepared_episodes_in_order(
            _record_iter(),
            preprocess_workers=max(1, args.preprocess_workers),
            max_in_flight=max(1, args.max_prepared_in_flight),
        ):
            cache_dir = Path(prepared.cache_dir)
            try:
                if (
                    not prepared.ok
                    or prepared.assets is None
                    or prepared.states is None
                    or prepared.pose3d_payload is None
                    or prepared.n_frames < 2
                ):
                    logger.warning(
                        "skip %s: %s",
                        prepared.record.get("id"),
                        prepared.error or "incomplete prepared episode",
                    )
                    total_failed += 1
                    continue

                ep_idx = dataset.meta.total_episodes
                _write_episode_to_dataset(
                    dataset, prepared.assets, prepared.n_frames, prepared.states,
                )
                ann = build_annotation(
                    prepared.record, prepared.assets,
                    final_length=prepared.n_frames,
                    source_fps=prepared.source_fps,
                    pose3d_payload=prepared.pose3d_payload,
                    cam_motion_score=prepared.cam_motion,
                )
                with open(annotations_dir / f"episode_{ep_idx:06d}.json", "w") as f:
                    json.dump(ann, f, ensure_ascii=False)
                total_done += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("episode failed (id=%s): %s",
                                 prepared.record.get("id"), exc)
                total_failed += 1
                try:
                    dataset.episode_buffer = dataset.create_episode_buffer()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=True)

            now = time.time()
            if now - last_log >= 60:
                logger.info("[%s] done=%d failed=%d elapsed=%.0fs",
                            args.subset_name, total_done, total_failed, now - t0)
                last_log = now
    finally:
        try:
            dataset.finalize()
        except Exception as exc:  # noqa: BLE001
            logger.exception("finalize failed: %s", exc)

    if total_done == 0:
        logger.warning("[%s] produced 0 episodes (failed=%d), leaving output empty",
                       args.subset_name, total_failed)
        shutil.rmtree(work_root, ignore_errors=True)
        return 0

    # Move local dataset_root -> output_root via plain `cp -r` (data only,
    # no metadata copy) so ossfs doesn't reject chown on cross-device move.
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cp", "-r", str(dataset_root), str(output_root)])
    shutil.rmtree(work_root, ignore_errors=True)

    logger.info(
        "[%s] DONE done=%d failed=%d elapsed=%.1fs output=%s",
        args.subset_name, total_done, total_failed, time.time() - t0, output_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
