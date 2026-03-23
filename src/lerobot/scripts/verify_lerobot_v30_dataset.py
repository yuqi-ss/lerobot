#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Load a local LeRobot v3.0 dataset and sanity-check metadata + a few frames.

Example:
  python src/lerobot/scripts/verify_lerobot_v30_dataset.py \\
    --root /path/to/database_lerobot_00_v30 \\
    --repo-id database_lerobot_00_v30 \\
    --num-samples 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def _summarize_tensor(x, name: str, max_elems: int = 4) -> str:
    if isinstance(x, torch.Tensor):
        flat = x.detach().cpu().flatten()
        n = min(flat.numel(), max_elems)
        head = flat[:n].tolist()
        return f"{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype} head={head!r}"
    return f"{name}: {type(x).__name__} = {x!r}"


def main() -> int:
    init_logging()
    parser = argparse.ArgumentParser(description="Verify a local LeRobot v3.0 dataset tree.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root (contains meta/, data/, optional videos/).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Logical repo id (defaults to the last component of --root).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="How many frame indices to load via __getitem__ (spread across the dataset).",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Comma-separated frame indices to load instead of spreading (e.g. 0,100,200).",
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Allow hub download for missing video files (default: local files only).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 1

    repo_id = args.repo_id or root.name
    print(f"repo_id={repo_id!r}  root={root}")

    # --- Metadata ---
    try:
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(root))
    except Exception as e:
        print(f"ERROR: LeRobotDatasetMetadata failed: {e}", file=sys.stderr)
        return 1

    ver = meta.info.get("codebase_version", "?")
    print(f"codebase_version={ver!r} (expected {CODEBASE_VERSION!r})")
    if ver != CODEBASE_VERSION:
        print("WARNING: codebase_version does not match v3.0.", file=sys.stderr)

    print(f"fps={meta.fps}  total_episodes={meta.total_episodes}  total_frames={meta.total_frames}")
    print(f"video_keys={meta.video_keys!r}")
    print(f"camera_keys={meta.camera_keys!r}")
    print(f"features={list(meta.features.keys())}")

    # --- Dataset samples ---
    try:
        ds = LeRobotDataset(
            repo_id=repo_id,
            root=str(root),
            download_videos=args.download_videos,
        )
    except Exception as e:
        print(f"ERROR: LeRobotDataset construction failed: {e}", file=sys.stderr)
        return 1

    n = len(ds)
    print(f"len(dataset)={n} (num_frames)")

    if n == 0:
        print("ERROR: dataset has zero frames.", file=sys.stderr)
        return 1

    if args.indices:
        try:
            idxs = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        except ValueError as e:
            print(f"ERROR: bad --indices: {e}", file=sys.stderr)
            return 1
    else:
        k = max(1, min(args.num_samples, n))
        if k == 1:
            idxs = [0]
        else:
            step = max(1, (n - 1) // (k - 1))
            idxs = sorted({min(n - 1, i * step) for i in range(k)})

    print(f"Loading frame indices: {idxs}")

    for i in idxs:
        if i < 0 or i >= n:
            print(f"ERROR: index {i} out of range [0, {n})", file=sys.stderr)
            return 1
        try:
            item = ds[i]
        except Exception as e:
            print(f"ERROR: ds[{i}] failed: {e}", file=sys.stderr)
            return 1

        print(f"--- sample index={i} ---")
        print(f"  keys={sorted(item.keys())}")
        for key in sorted(item.keys()):
            val = item[key]
            if isinstance(val, torch.Tensor):
                print(" ", _summarize_tensor(val, key))
            elif isinstance(val, str):
                s = val if len(val) < 120 else val[:117] + "..."
                print(f"  {key}: str len={len(val)} {s!r}")
            else:
                print(f"  {key}: {type(val).__name__}")

    print("OK: dataset loads and samples succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# python src/lerobot/scripts/verify_lerobot_v30_dataset.py \
#   --root /home/ss-oss1/data/dataset/egocentric/training_egocentric/retarget_lerobot/database_lerobot_00_v30 \
#   --repo-id database_lerobot_00 \
#   --num-samples 5