"""Convert egocentric_dataset_clips (Ego4D / Egocentric-100K partitions)
from Supabase directly to LeRobot v3.0 datasets.

Layout follows the existing mjc v2.1 pipeline 1:1 (chunk-NNNNNN/ subdataset
+ side-loaded annotations/), with the following deltas:

  * codebase_version is v3.0 (LeRobotDataset.create)
  * video container stays at 30fps but only every 10th source frame is
    actually decoded; the 9 in between repeat the latest decoded frame.
    svtav1 compresses repeated frames extremely well, so the on-disk mp4
    is roughly the size of a true 3fps stream.
  * IK/state/action remain 30Hz; if the source IK fps is not 30, values
    are linearly resampled (continuous) / nearest-neighbour (bool) onto a
    uniform 30Hz timeline.
  * annotation JSON gains:
      - caption: full {semantic_motion, short_caption, motion_sequence}
      - pose3d_hand: per-frame mano + relative_motion (no rel_rot_mat)
                     and slam_data without the heavy disps tensor
      - camera_motion_score: mean per-frame translation of slam_data.traj
      - video_fps: 3 (logical visual frame rate)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

# Local DB helpers (same package).
from lerobot.scripts.egocentric_database_operation import (  # noqa: E402
    EGOCENTRIC_DATASET_CLIPS_TABLE,
    PARTITION_EGO100K,
    PARTITION_EGO4D,
    get_egocentric_boundaries,
    get_egocentric_paths_from_database,
    supabase,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TARGET_FPS = 30
VIDEO_DOWNSAMPLE = 10  # decode 1 source frame every 10 timeline frames -> 3Hz visual
VIDEO_KEY = "observation.images.cam_high"  # match mjc v2.1 naming

# Match data_transform/convert_database_to_lerobot_video.py for video.
TARGET_VIDEO_H = 480
TARGET_VIDEO_W = 640
# EEF (end-effector) 16-D state/action layout:
#   [0:3]   left_pos        = left_pose[0:3]
#   [3:7]   left_quat xyzw  = axis_angle_to_quat(left_pose[3:6])
#   [7]     left_gripper_signal
#   [8:11]  right_pos       = right_pose[0:3]
#   [11:15] right_quat xyzw = axis_angle_to_quat(right_pose[3:6])
#   [15]    right_gripper_signal
STATE_DIM = 16
ACTION_DIM = STATE_DIM  # state == action by design


# -----------------------------------------------------------------------------
# Path translation: oss:// -> /home/
# -----------------------------------------------------------------------------
def to_local_path(p: str | None) -> str | None:
    if not p:
        return p
    if p.startswith("oss://"):
        return "/home/" + p[len("oss://") :]
    return p


# -----------------------------------------------------------------------------
# 16-D EEF state/action assembly (see STATE_DIM comment for layout).
# -----------------------------------------------------------------------------
def _axis_angle_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """rotvec: (3,) -> quat (4,) in (x, y, z, w) order."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(-1)
    if rotvec.shape[0] != 3:
        raise ValueError(f"axis-angle must be length 3, got shape {rotvec.shape}")
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = rotvec / angle
    half = angle * 0.5
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)], dtype=np.float32)


def _eef_pose6_to_pos_quat(pose6: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 6-D pose (xyz + axis-angle) to (pos[3], quat_xyzw[4]).

    Raises ValueError if the input is missing / wrong shape, so the caller
    can skip the offending episode without crashing the chunk.
    """
    p = np.asarray(pose6, dtype=np.float32).reshape(-1)
    if p.shape[0] != 6:
        raise ValueError(f"pose must be length 6 (xyz + axis-angle), got shape {p.shape}")
    return p[0:3].astype(np.float32), _axis_angle_to_quat_xyzw(p[3:6])


def _build_state_16d(ik_row: dict) -> np.ndarray:
    if "left_pose" not in ik_row or "right_pose" not in ik_row:
        raise ValueError("left_pose/right_pose missing from ik row")
    l_pos, l_quat = _eef_pose6_to_pos_quat(ik_row["left_pose"])
    r_pos, r_quat = _eef_pose6_to_pos_quat(ik_row["right_pose"])
    out = np.empty((STATE_DIM,), dtype=np.float32)
    out[0:3] = l_pos
    out[3:7] = l_quat
    out[7] = float(ik_row["left_gripper_signal"])
    out[8:11] = r_pos
    out[11:15] = r_quat
    out[15] = float(ik_row["right_gripper_signal"])
    return out


# -----------------------------------------------------------------------------
# IK parquet -> resampled 30Hz dict of per-frame numpy arrays
# -----------------------------------------------------------------------------
def _resample_ik_to_30hz(ik_df, source_fps: float) -> tuple[list[np.ndarray], int]:
    """Return list of 16-D EEF state vectors (one per target 30Hz frame).

    If source_fps == 30 (the common case) this is a straight per-row build.
    Otherwise continuous fields are linearly interpolated on the time axis.
    Raises ValueError if any IK row is malformed (caller should skip the ep).
    """
    n_src = len(ik_df)
    if abs(source_fps - TARGET_FPS) < 1e-6 or n_src <= 1:
        states = [_build_state_16d(ik_df.iloc[i]) for i in range(n_src)]
        return states, n_src

    duration = (n_src - 1) / source_fps
    n_tgt = int(round(duration * TARGET_FPS)) + 1
    src_t = np.arange(n_src) / source_fps
    tgt_t = np.arange(n_tgt) / TARGET_FPS

    src_states = np.stack([_build_state_16d(ik_df.iloc[i]) for i in range(n_src)], axis=0)
    out = np.empty((n_tgt, STATE_DIM), dtype=np.float32)
    for d in range(STATE_DIM):
        out[:, d] = np.interp(tgt_t, src_t, src_states[:, d])
    return [out[i] for i in range(n_tgt)], n_tgt


# -----------------------------------------------------------------------------
# pose3d_hand zip parser (PyTorch zip)
# -----------------------------------------------------------------------------
def _load_pose3d_hand(path: str) -> dict:
    import torch

    return torch.load(path, weights_only=False, map_location="cpu")


def _to_listy(x):
    """Recursively convert torch tensors / numpy arrays to plain Python lists."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_listy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_listy(v) for v in x]
    return x


def _build_pose3d_hand_payload(raw: dict) -> dict:
    """Drop slam_data.disps and *.relative_motion.rel_rot_mat to slim down."""
    out: dict[str, Any] = {"fps": float(raw.get("fps", TARGET_FPS))}
    for side in ("left_hand", "right_hand"):
        h = raw.get(side, {}) or {}
        mano = h.get("mano_params", {}) or {}
        rm = h.get("relative_motion", {}) or {}
        out[side] = {
            "mano_params": {
                "global_orient": _to_listy(mano.get("global_orient")),
                "hand_pose":     _to_listy(mano.get("hand_pose")),
                "betas":         _to_listy(mano.get("betas")),
                "transl":        _to_listy(mano.get("transl")),
            },
            "pred_valid": _to_listy(h.get("pred_valid")),
            "relative_motion": {
                "rel_rot_aa": _to_listy(rm.get("rel_rot_aa")),
                "rel_trans":  _to_listy(rm.get("rel_trans")),
                "pair_valid": _to_listy(rm.get("pair_valid")),
            },
        }
    slam = raw.get("slam_data", {}) or {}
    out["slam_data"] = {
        "tstamp":     _to_listy(slam.get("tstamp")),
        "traj":       _to_listy(slam.get("traj")),
        "img_focal":  _to_listy(slam.get("img_focal")),
        "img_center": _to_listy(slam.get("img_center")),
        "scale":      _to_listy(slam.get("scale")),
    }
    return out


def _camera_motion_score(slam_traj: Any) -> float:
    """Mean per-frame translation magnitude. slam_traj: (T,7) -> float."""
    if slam_traj is None:
        return float("nan")
    arr = np.asarray(slam_traj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
        return float("nan")
    delta = arr[1:, :3] - arr[:-1, :3]
    return float(np.linalg.norm(delta, axis=-1).mean())


# -----------------------------------------------------------------------------
# Video reader (downsamples to ~3Hz by repeating the last decoded frame)
# -----------------------------------------------------------------------------
def _open_video(path: str):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    return cap


def _iter_video_frames_downsampled(
    video_path: str,
    n_frames: int,
    downsample: int = VIDEO_DOWNSAMPLE,
    target_h: int = TARGET_VIDEO_H,
    target_w: int = TARGET_VIDEO_W,
) -> Iterator[np.ndarray]:
    """Yield exactly `n_frames` RGB uint8 (target_h, target_w, 3) arrays.

    Decodes only frames whose index % downsample == 0; in-between frames
    are repeats of the latest decoded frame. All frames are resized to
    (target_h, target_w) so the LeRobot v3.0 dataset has a uniform shape.
    """
    import cv2

    cap = _open_video(video_path)
    last: np.ndarray | None = None
    src_idx = 0
    try:
        for tgt_idx in range(n_frames):
            need_decode = (tgt_idx % downsample == 0) or last is None
            if need_decode:
                while src_idx <= tgt_idx:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    src_idx += 1
                if frame is None:
                    if last is None:
                        raise RuntimeError(f"empty video: {video_path}")
                    yield last
                    continue
                if frame.shape[1] != target_w or frame.shape[0] != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last = np.ascontiguousarray(rgb, dtype=np.uint8)
            yield last
    finally:
        cap.release()


# -----------------------------------------------------------------------------
# Per-clip loader
# -----------------------------------------------------------------------------
@dataclass
class ClipAssets:
    record: dict
    ik_meta: dict
    caption: dict
    pose3d_raw: dict
    video_path: str
    ik_data_path: str
    pose_path: str
    video_height: int
    video_width: int


@dataclass
class PreparedEpisode:
    ok: bool
    record: dict
    cache_dir: str
    assets: ClipAssets | None = None
    states: list[np.ndarray] | None = None
    n_frames: int = 0
    source_fps: float = TARGET_FPS
    pose3d_payload: dict | None = None
    cam_motion: float = float("nan")
    error: str | None = None


def _parse_caption(raw: Any) -> dict:
    if isinstance(raw, dict):
        d = raw
    elif isinstance(raw, str):
        try:
            d = json.loads(raw)
        except Exception:
            d = {"short_caption": raw}
    else:
        d = {}
    return {
        "semantic_motion": d.get("semantic_motion") or "",
        "short_caption":   d.get("short_caption") or "",
        "motion_sequence": d.get("motion_sequence") or "",
    }


def _probe_video_resolution(path: str) -> tuple[int, int]:
    import cv2

    cap = _open_video(path)
    try:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    finally:
        cap.release()
    return h, w


def _stage_record_files(
    record: dict, cache_dir: Path
) -> tuple[str, str, str, str] | None:
    """Copy source artifacts from ossfs to a per-record local cache directory.

    ffmpeg/cv2/parquet/torch.load all want random-access reads; on ossfs every
    seek triggers an OSS RANGE request and dominates wall time.  Doing one
    sequential `cp` (ossfs reads ~370 MB/s) per file before processing turns
    seek-heavy access into a local NVMe operation (microseconds).

    Returns local (video, ik_data, ik_meta, pose3d) paths, or None if any
    source is missing.  Caller is responsible for cleaning up cache_dir.
    """
    sources = {
        "video":   to_local_path(record.get("path")),
        "ik_data": to_local_path(record.get("agilex_ik_result_data_path")),
        "ik_meta": to_local_path(record.get("agilex_ik_result_meta_data_path")),
        "pose3d":  to_local_path(record.get("pose3d_hand_path")),
    }
    for label, src in sources.items():
        if not src or not os.path.exists(src):
            logger.warning("skip %s: %s missing (%s)", record.get("id"), label, src)
            return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = {}
    for label, src in sources.items():
        suffix = Path(src).suffix or ""
        dst = cache_dir / f"{label}{suffix}"
        # plain os.cpy: only data, no chmod/chown -> fast and OSS-safe.
        with open(src, "rb") as fr, open(dst, "wb") as fw:
            shutil.copyfileobj(fr, fw, length=4 * 1024 * 1024)
        local[label] = str(dst)
    return local["video"], local["ik_data"], local["ik_meta"], local["pose3d"]


def load_clip_assets(record: dict, cache_dir: Path | None = None) -> ClipAssets | None:
    """Return assets for one DB record, or None if any required artifact is missing.

    If ``cache_dir`` is given, source files are first staged there so ffmpeg/cv2
    don't pay ossfs FUSE seek latency on every frame decode.
    """
    if cache_dir is not None:
        staged = _stage_record_files(record, cache_dir)
        if staged is None:
            return None
        video_path, ik_data_path, ik_meta_path, pose_path = staged
    else:
        video_path = to_local_path(record.get("path"))
        ik_data_path = to_local_path(record.get("agilex_ik_result_data_path"))
        ik_meta_path = to_local_path(record.get("agilex_ik_result_meta_data_path"))
        pose_path = to_local_path(record.get("pose3d_hand_path"))
        for label, p in [("video", video_path), ("ik_data", ik_data_path),
                         ("ik_meta", ik_meta_path), ("pose3d_hand", pose_path)]:
            if not p or not os.path.exists(p):
                logger.warning("skip %s: %s missing (%s)", record.get("id"), label, p)
                return None

    with open(ik_meta_path) as f:
        ik_meta = json.load(f)
    caption = _parse_caption(record.get("caption"))
    pose3d_raw = _load_pose3d_hand(pose_path)
    h, w = _probe_video_resolution(video_path)

    return ClipAssets(
        record=record,
        ik_meta=ik_meta,
        caption=caption,
        pose3d_raw=pose3d_raw,
        video_path=video_path,
        ik_data_path=ik_data_path,
        pose_path=pose_path,
        video_height=h,
        video_width=w,
    )


def _prepare_episode_worker(record: dict, cache_dir_str: str) -> PreparedEpisode:
    """Stage source files and prepare CPU-heavy per-episode payloads.

    The caller must still write to LeRobotDataset in the parent process because
    dataset metadata, episode indices, and streaming encoders are not
    multiprocess-safe.
    """
    cache_dir = Path(cache_dir_str)
    try:
        import pyarrow.parquet as pq

        assets = load_clip_assets(record, cache_dir=cache_dir)
        if assets is None:
            return PreparedEpisode(
                ok=False,
                record=record,
                cache_dir=cache_dir_str,
                error="missing_assets",
            )

        ik_df = pq.read_table(assets.ik_data_path).to_pandas()
        source_fps = float(assets.ik_meta.get("fps", TARGET_FPS))
        states, n_frames = _resample_ik_to_30hz(ik_df, source_fps)
        if n_frames < 2:
            return PreparedEpisode(
                ok=False,
                record=record,
                cache_dir=cache_dir_str,
                assets=assets,
                error=f"only {n_frames} frames after resample",
            )

        pose3d_payload = _build_pose3d_hand_payload(assets.pose3d_raw)
        cam_motion = _camera_motion_score(pose3d_payload["slam_data"]["traj"])
        # Avoid pickling the raw torch payload back to the parent process. The
        # compact JSON-ready payload above is all the writer needs.
        assets.pose3d_raw = {}
        return PreparedEpisode(
            ok=True,
            record=record,
            cache_dir=cache_dir_str,
            assets=assets,
            states=states,
            n_frames=n_frames,
            source_fps=source_fps,
            pose3d_payload=pose3d_payload,
            cam_motion=cam_motion,
        )
    except Exception as exc:  # noqa: BLE001
        return PreparedEpisode(
            ok=False,
            record=record,
            cache_dir=cache_dir_str,
            error=f"{type(exc).__name__}: {exc}",
        )


def _iter_prepared_episodes_in_order(
    items: Iterator[tuple[dict, Path]],
    *,
    preprocess_workers: int,
    max_in_flight: int,
) -> Iterator[PreparedEpisode]:
    """Yield prepared episode payloads in input order."""
    if preprocess_workers <= 1:
        for record, cache_dir in items:
            yield _prepare_episode_worker(record, str(cache_dir))
        return

    next_submit_index = 0
    next_yield_index = 0
    iterator = iter(items)
    futures: dict[concurrent.futures.Future[PreparedEpisode], int] = {}
    completed: dict[int, PreparedEpisode] = {}

    def submit_one(executor: concurrent.futures.ProcessPoolExecutor) -> bool:
        nonlocal next_submit_index
        try:
            record, cache_dir = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(_prepare_episode_worker, record, str(cache_dir))
        futures[future] = next_submit_index
        next_submit_index += 1
        return True

    with concurrent.futures.ProcessPoolExecutor(max_workers=preprocess_workers) as executor:
        while len(futures) < max_in_flight and submit_one(executor):
            pass

        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                idx = futures.pop(future)
                completed[idx] = future.result()

            while next_yield_index in completed:
                yield completed.pop(next_yield_index)
                next_yield_index += 1
                while len(futures) < max_in_flight and submit_one(executor):
                    pass


# -----------------------------------------------------------------------------
# Annotation builder (mirrors mjc v2.1 layout + 4 new top-level fields)
# -----------------------------------------------------------------------------
def build_annotation(
    record: dict,
    assets: ClipAssets,
    final_length: int,
    source_fps: float,
    pose3d_payload: dict,
    cam_motion_score: float,
) -> dict:
    short_caption = assets.caption.get("short_caption") or ""

    manifest_entry = {
        "table_name": EGOCENTRIC_DATASET_CLIPS_TABLE,
        "table_alias": "ego4d" if record.get("partition") == PARTITION_EGO4D else "ego100k",
        "partition": record.get("partition"),
        "dataset_name": record.get("dataset_name"),
        "sub": record.get("sub"),
        "part": record.get("part"),
        "sample_id": record.get("id"),
        "text": short_caption,
        "video_path": assets.video_path,
        "pose_path": assets.pose_path,
        "ik_data_path": assets.ik_data_path,
        "ik_meta_path": to_local_path(record.get("agilex_ik_result_meta_data_path")),
        "source_path": to_local_path(record.get("source_path")),
        "start_frame": record.get("start_frame"),
        "end_frame": record.get("end_frame"),
        "fps": record.get("fps"),
        "height_resolution": record.get("height_resolution"),
        "width_resolution": record.get("width_resolution"),
        "hand_tag": record.get("hand_tag"),
        "motion_score": record.get("motion_score"),
    }

    return {
        "manifest_entry": manifest_entry,
        "ik_meta": assets.ik_meta,
        "caption": assets.caption,
        "pose3d_hand": pose3d_payload,
        "camera_motion_score": cam_motion_score,
        "video_fps": TARGET_FPS // VIDEO_DOWNSAMPLE,
        "final_length": final_length,
        "video_height": TARGET_VIDEO_H,
        "video_width": TARGET_VIDEO_W,
        "source_video_height": assets.video_height,
        "source_video_width": assets.video_width,
        "fps": float(TARGET_FPS),
        "source_fps": float(source_fps),
        "temporal_downsample": 1,
    }


# -----------------------------------------------------------------------------
# Per-chunk LeRobot v3.0 builder
# -----------------------------------------------------------------------------
def _features_dict(video_h: int = TARGET_VIDEO_H, video_w: int = TARGET_VIDEO_W) -> dict:
    # NOTE: shapes MUST be tuples — LeRobot validates `np.ndarray.shape == feature["shape"]`
    # which is `tuple == list` -> always False if we use lists here.
    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (3, video_h, video_w),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action":            {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }


def _open_chunk_dataset(
    repo_id: str, root: Path,
    video_h: int = TARGET_VIDEO_H, video_w: int = TARGET_VIDEO_W,
    vcodec: str = "libsvtav1",
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=TARGET_FPS,
        features=_features_dict(video_h, video_w),
        root=root,
        robot_type="egocentric_eef",
        use_videos=True,
        # Stream frames directly into ffmpeg encoder threads (no PNG round-trip);
        # video stats are produced by the encoder, so we never read PNGs back.
        streaming_encoding=True,
        encoder_queue_maxsize=256,
        vcodec=vcodec,
        batch_encoding_size=1,
    )


def _write_episode_to_dataset(dataset, assets: ClipAssets, n_frames: int, states: list[np.ndarray]) -> None:
    task_str = assets.caption.get("short_caption") or "egocentric clip"
    frame_iter = _iter_video_frames_downsampled(assets.video_path, n_frames)
    for k, img in enumerate(frame_iter):
        if k >= n_frames:
            break
        st = states[k]
        dataset.add_frame({
            VIDEO_KEY: img,
            "observation.state": st,
            "action": st.copy(),
            "task": task_str,
        })
    dataset.save_episode()


# -----------------------------------------------------------------------------
# DB iterator with simple resume support
# -----------------------------------------------------------------------------
def _count_records(
    partition: str,
    path_gte: str | None,
    path_lt: str | None,
) -> int | None:
    """Best-effort exact count of records that will be yielded by `_iter_records`.

    Mirrors the same filters (partition + path range + non-null required paths).
    Returns None on failure so the caller can fall back to an unbounded progress bar.
    """
    try:
        q = (
            supabase.table(EGOCENTRIC_DATASET_CLIPS_TABLE)
            .select("path", count="exact")
            .limit(1)
            .eq("partition", partition)
            .not_.is_("agilex_ik_result_data_path", "null")
            .not_.is_("agilex_ik_result_meta_data_path", "null")
            .not_.is_("pose3d_hand_path", "null")
        )
        if path_gte is not None:
            q = q.gte("path", path_gte)
        if path_lt is not None:
            q = q.lt("path", path_lt)
        return q.execute().count or 0
    except Exception as e:  # noqa: BLE001
        logger.warning("count_records failed (%s); progress bar will be unbounded", e)
        return None


def _iter_records(
    partition: str,
    path_gte: str | None,
    path_lt: str | None,
    page_size: int,
    max_episodes: int | None,
) -> Iterator[dict]:
    """Yield records that have all required artifacts (paths non-null in DB)."""
    select_fields = [
        "path", "id", "partition", "dataset_name", "sub", "part",
        "source_path", "start_frame", "end_frame",
        "fps", "height_resolution", "width_resolution",
        "hand_tag", "motion_score",
        "caption",
        "agilex_ik_result_data_path",
        "agilex_ik_result_meta_data_path",
        "pose3d_hand_path",
    ]
    res = get_egocentric_paths_from_database(
        partition=partition,
        path_gte=path_gte,
        path_lt=path_lt,
        select_fields=select_fields,
        filter_empty_fields=[
            "agilex_ik_result_data_path",
            "agilex_ik_result_meta_data_path",
            "pose3d_hand_path",
        ],
        page_size=page_size,
        total_num=max_episodes,
    )
    yield from (res.data or [])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", required=True, choices=[PARTITION_EGO4D, PARTITION_EGO100K])
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--episodes-per-chunk", type=int, default=5000)
    p.add_argument("--start-chunk-index", type=int, default=0)
    p.add_argument("--shard", type=int, default=None, help="0-based shard id when using --num-shards")
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--path-gte", type=str, default=None)
    p.add_argument("--path-lt", type=str, default=None)
    p.add_argument("--max-episodes", type=int, default=None, help="debug cap, overall")
    p.add_argument("--page-size", type=int, default=200)
    p.add_argument("--preprocess-workers", type=int, default=1)
    p.add_argument("--max-prepared-in-flight", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--vcodec", default="libsvtav1",
        help="ffmpeg/PyAV video encoder; e.g. libsvtav1 (CPU, default), h264_nvenc / hevc_nvenc (GPU).",
    )
    p.add_argument(
        "--work-dir", type=Path, default=Path("/home/yuqi/tmp/egocentric_v30_work"),
        help=(
            "Local scratch directory where each chunk is built before it is moved "
            "to --output-root. Required when --output-root sits on ossfs/network FS "
            "that does not support ffmpeg's random/append writes (errno 22)."
        ),
    )
    return p.parse_args()


def _resolve_path_range(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.path_gte is not None or args.path_lt is not None:
        return args.path_gte, args.path_lt
    if args.shard is not None and args.num_shards is not None:
        # Sample interval must yield at least num_shards+1 boundary points.
        # Ego4D ~240k rows: with 12 shards we need ~13 points, so cap interval << 240k/13.
        sample_interval = max(500, 200_000 // max(1, args.num_shards))
        boundaries = get_egocentric_boundaries(
            partition=args.partition,
            total_machines=args.num_shards,
            sample_interval=sample_interval,
            verbose=False,
        )
        if args.shard < 0 or args.shard >= args.num_shards:
            raise SystemExit(f"shard {args.shard} out of range [0,{args.num_shards})")
        if len(boundaries) < args.num_shards + 1:
            raise SystemExit(f"could not derive enough boundaries: got {len(boundaries)}")
        return boundaries[args.shard], boundaries[args.shard + 1]
    return None, None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Silence per-request HTTP logs that otherwise flood the terminal during
    # large-scale paginated reads / resume-skip.
    for noisy in ("httpx", "httpcore", "hpack", "urllib3", "postgrest"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Silence ffmpeg/PyAV stderr completely. ERROR(=16) still leaks the
    # `[mp4 @ ...] Starting second pass` mov-muxer line; PANIC(=0) drops it.
    try:
        import av
        # Both APIs needed: set_level affects PyAV's Python logger;
        # set_libav_level affects the underlying ffmpeg C av_log_level
        # (which is what produces "[mp4 @ 0x...] Starting second pass" etc).
        av.logging.set_level(av.logging.PANIC)
        av.logging.set_libav_level(av.logging.PANIC)
    except Exception:  # noqa: BLE001
        pass
    os.environ.setdefault("SVT_LOG", "1")            # 1 = errors only
    os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
    # Disable huggingface datasets' Map: 100%|...| progress bars (LeRobot's
    # internal create_hf_dataset / _save_episode_data calls ds.map() per ep).
    try:
        import datasets as _hfds
        _hfds.disable_progress_bar()
    except Exception:  # noqa: BLE001
        pass
    if args.overwrite and args.output_root.exists():
        logger.warning("--overwrite: wiping %s", args.output_root)
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Per-shard local scratch under --work-dir; LeRobot's streaming encoder writes
    # ffmpeg temp .mp4 here, so this MUST be a real local POSIX FS (not ossfs).
    shard_tag = f"shard{args.shard if args.shard is not None else 0}"
    work_root = args.work_dir / f"{args.partition.lower().replace('-', '_')}_{shard_tag}"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    # Local scratch sanity-check. At 640x480 h264_nvenc / libsvtav1 a single ep
    # mp4 measures ~3.3 MB; one shard keeps the whole chunk in flight before
    # cp->remote+rm cleanup runs. Source-cache adds ~50 MB transient per record.
    free_gb = shutil.disk_usage(args.work_dir).free / (1 << 30)
    est_chunk_gb = args.episodes_per_chunk * 0.0035 + 0.1  # 3.5 MB/ep + cache slack
    if est_chunk_gb > free_gb * 0.8:
        logger.warning(
            "local scratch %s has only %.1f GB free, single-chunk peak ~%.1f GB "
            "(episodes_per_chunk=%d). Consider lowering --episodes-per-chunk.",
            args.work_dir, free_gb, est_chunk_gb, args.episodes_per_chunk,
        )
    path_gte, path_lt = _resolve_path_range(args)

    logger.info("partition=%s shard_range=[%s, %s) episodes_per_chunk=%s output=%s work=%s",
                args.partition, path_gte, path_lt, args.episodes_per_chunk,
                args.output_root, work_root)

    chunk_idx = args.start_chunk_index
    chunk_dataset = None
    chunk_root: Path | None = None       # local scratch path being written to
    chunk_remote: Path | None = None     # final destination on output_root
    chunk_count = 0  # episodes saved into the current chunk
    seen_in_chunk = 0  # records consumed (incl. skipped) for current chunk
    total_done = 0
    total_failed = 0
    total_skipped_resume = 0
    t0 = time.time()

    def _chunk_dir_name(idx: int) -> str:
        return f"chunk-{idx:06d}_v30"

    def _is_chunk_complete(idx: int) -> bool:
        # A chunk is considered complete only after LeRobot finalize() has
        # written meta/info.json + meta/episodes/*.parquet for it.
        cr = args.output_root / _chunk_dir_name(idx)
        return (cr / "meta" / "info.json").exists() and (cr / "meta" / "episodes").is_dir()

    chunk_complete = _is_chunk_complete(chunk_idx)
    if chunk_complete:
        logger.info("resume: chunk-%06d already complete, skipping its records", chunk_idx)

    total_records = _count_records(args.partition, path_gte, path_lt)
    if args.max_episodes is not None:
        total_records = min(total_records or args.max_episodes, args.max_episodes)
    logger.info("estimated %s records to process", total_records if total_records is not None else "?")
    pbar = tqdm(
        total=total_records,
        unit="ep",
        smoothing=0.05,
        dynamic_ncols=True,
        desc=f"{args.partition}",
    )

    def _refresh_pbar():
        pbar.set_postfix(
            done=total_done, failed=total_failed, skipped=total_skipped_resume,
            chunk=f"{chunk_idx:06d}({chunk_count}/{args.episodes_per_chunk})",
            refresh=False,
        )

    def _close_chunk():
        """Finalize current chunk locally and move it to args.output_root."""
        nonlocal chunk_dataset, chunk_root, chunk_remote
        if chunk_dataset is not None:
            try:
                chunk_dataset.finalize()
            except Exception as e:
                logger.exception("finalize failed for chunk %s: %s", chunk_idx, e)
            chunk_dataset = None
        if chunk_root is not None and chunk_root.exists() and chunk_remote is not None:
            try:
                if chunk_remote.exists():
                    shutil.rmtree(chunk_remote)
                chunk_remote.parent.mkdir(parents=True, exist_ok=True)
                # Cross-device move: shutil.move falls back to copy2 which tries to
                # preserve owner/mode -- ossfs rejects chown -> Operation not permitted.
                # Use plain `cp -r` (data only, no metadata) then rm the local copy.
                import subprocess
                subprocess.check_call(["cp", "-r", str(chunk_root), str(chunk_remote)])
                shutil.rmtree(chunk_root)
                logger.info("moved chunk-%06d local->remote: %s", chunk_idx, chunk_remote)
            except Exception as e:
                logger.exception("move local->remote failed for chunk %s: %s", chunk_idx, e)
        chunk_root = None
        chunk_remote = None

    def _advance_chunk():
        """Move to the next chunk index and refresh resume state."""
        nonlocal chunk_idx, chunk_count, seen_in_chunk, chunk_complete
        chunk_idx += 1
        chunk_count = 0
        seen_in_chunk = 0
        chunk_complete = _is_chunk_complete(chunk_idx)
        if chunk_complete:
            logger.info("resume: chunk-%06d already complete, skipping its records", chunk_idx)

    preprocess_workers = max(1, int(args.preprocess_workers))
    max_prepared_in_flight = args.max_prepared_in_flight
    if max_prepared_in_flight is None:
        max_prepared_in_flight = max(1, preprocess_workers * 2)
    max_prepared_in_flight = max(1, int(max_prepared_in_flight))
    logger.info(
        "preprocess_workers=%d max_prepared_in_flight=%d",
        preprocess_workers,
        max_prepared_in_flight,
    )

    record_seq = 0

    def _active_record_items() -> Iterator[tuple[dict, Path]]:
        nonlocal seen_in_chunk, total_skipped_resume, record_seq
        for record in _iter_records(args.partition, path_gte, path_lt, args.page_size, args.max_episodes):
            pbar.update(1)
            # ── resume: silently consume records belonging to a finished chunk ──
            if chunk_complete:
                seen_in_chunk += 1
                total_skipped_resume += 1
                if seen_in_chunk >= args.episodes_per_chunk:
                    _advance_chunk()
                _refresh_pbar()
                continue

            # Per-record local cache: stage all source artefacts off ossfs so
            # ffmpeg/cv2/torch.load do their seek-heavy work on local NVMe.
            cache_dir = work_root / "_src_cache" / f"rec_{record_seq:08d}_{str(record.get('id', ''))[:12]}"
            record_seq += 1
            yield record, cache_dir

    try:
        prepared_iter = _iter_prepared_episodes_in_order(
            _active_record_items(),
            preprocess_workers=preprocess_workers,
            max_in_flight=max_prepared_in_flight,
        )
        for prepared in prepared_iter:
            record = prepared.record
            cache_dir = Path(prepared.cache_dir)
            try:
                if not prepared.ok:
                    logger.warning("skip %s: %s", record.get("id"), prepared.error)
                    total_failed += 1
                    seen_in_chunk += 1
                    continue
                if (
                    prepared.assets is None
                    or prepared.states is None
                    or prepared.pose3d_payload is None
                    or prepared.n_frames < 2
                ):
                    logger.warning("skip %s: incomplete prepared episode", record.get("id"))
                    total_failed += 1
                    seen_in_chunk += 1
                    continue

                assets = prepared.assets
                if chunk_dataset is None:
                    chunk_remote = args.output_root / _chunk_dir_name(chunk_idx)
                    chunk_root = work_root / _chunk_dir_name(chunk_idx)
                    if chunk_remote.exists() and not args.overwrite:
                        # Partially-written remote chunk (no info.json yet): wipe and retry.
                        logger.warning(
                            "chunk-%06d remote exists but is incomplete, wiping for retry",
                            chunk_idx,
                        )
                        shutil.rmtree(chunk_remote)
                    if chunk_root.exists():
                        shutil.rmtree(chunk_root)
                    chunk_dataset = _open_chunk_dataset(
                        repo_id=f"chunk-{chunk_idx:06d}",
                        root=chunk_root,
                        vcodec=args.vcodec,
                    )
                    chunk_count = 0
                    logger.info(
                        "opened chunk-%06d at %s (video=%dx%d)",
                        chunk_idx,
                        chunk_root,
                        TARGET_VIDEO_H,
                        TARGET_VIDEO_W,
                    )

                try:
                    ep_idx = chunk_dataset.meta.total_episodes
                    _write_episode_to_dataset(
                        chunk_dataset,
                        assets,
                        prepared.n_frames,
                        prepared.states,
                    )

                    ann = build_annotation(
                        record,
                        assets,
                        final_length=prepared.n_frames,
                        source_fps=prepared.source_fps,
                        pose3d_payload=prepared.pose3d_payload,
                        cam_motion_score=prepared.cam_motion,
                    )
                    ann_dir = chunk_root / "annotations" / "chunk-000"
                    ann_dir.mkdir(parents=True, exist_ok=True)
                    with open(ann_dir / f"episode_{ep_idx:06d}.json", "w") as f:
                        json.dump(ann, f, ensure_ascii=False)

                    chunk_count += 1
                    total_done += 1
                    seen_in_chunk += 1
                except Exception as e:
                    logger.exception("episode failed (id=%s): %s", record.get("id"), e)
                    total_failed += 1
                    seen_in_chunk += 1
                    try:
                        if chunk_dataset is not None:
                            chunk_dataset.episode_buffer = chunk_dataset.create_episode_buffer()
                    except Exception:
                        pass
                    continue
            finally:
                # Always clean per-record cache, regardless of success/failure.
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=True)

            if chunk_count >= args.episodes_per_chunk:
                _close_chunk()
                _advance_chunk()

            _refresh_pbar()

    finally:
        _close_chunk()
        pbar.close()

    logger.info("DONE total_done=%d total_failed=%d resume_skipped=%d elapsed=%.1fs",
                total_done, total_failed, total_skipped_resume, time.time() - t0)


if __name__ == "__main__":
    sys.exit(main())
