"""Build Boston/Singapore scene-token splits from nuScenes mini.

Reads the nuScenes metadata, partitions scenes by `log.location`, and writes
a JSON of {boston_train, boston_val, singapore_train, singapore_val}.

The mini split has ~10 scenes total (typically ~6 Boston, ~4 Singapore).
Train/val ratio is configurable; defaults to 60/40 within each city.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable


NUSC_ROOT = "/kaggle/input/nuscenes-mini/v1.0-mini"
OUT_PATH = "/kaggle/working/splits/splits.json"
VERSION = "v1.0-mini"
TRAIN_RATIO = 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Core split logic — pure function over scene+log dicts so it's testable
# without nuscenes-devkit installed.
# ─────────────────────────────────────────────────────────────────────────────

def partition_by_location(
    scenes: Iterable[dict],
    logs_by_token: dict[str, dict],
    train_ratio: float = TRAIN_RATIO,
) -> dict[str, list[str]]:
    """Group scenes by city, then split each city into train/val.

    Args:
        scenes: iterable of scene records, each with at least
            {'token': str, 'log_token': str, 'name': str}.
        logs_by_token: map from log_token to log record with 'location' field.
        train_ratio: fraction of each city's scenes that go to train.

    Returns:
        Dict with keys boston_train, boston_val, singapore_train, singapore_val.
        Values are lists of scene tokens. Sorted by scene name for determinism.
    """
    boston: list[tuple[str, str]] = []     # (scene_name, scene_token)
    singapore: list[tuple[str, str]] = []

    for scene in scenes:
        log = logs_by_token.get(scene["log_token"])
        if log is None:
            continue
        loc = log.get("location", "")
        if loc.startswith("boston"):
            boston.append((scene["name"], scene["token"]))
        elif loc.startswith("singapore"):
            singapore.append((scene["name"], scene["token"]))

    boston.sort()
    singapore.sort()

    def _split(items: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        if not items:
            return [], []
        # Ensure at least 1 in val if there are >= 2 items.
        n_train = max(1, math.floor(len(items) * train_ratio))
        if len(items) >= 2:
            n_train = min(n_train, len(items) - 1)
        train = [tok for _, tok in items[:n_train]]
        val = [tok for _, tok in items[n_train:]]
        return train, val

    boston_train, boston_val = _split(boston)
    singapore_train, singapore_val = _split(singapore)

    return {
        "boston_train": boston_train,
        "boston_val": boston_val,
        "singapore_train": singapore_train,
        "singapore_val": singapore_val,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle entrypoint — wraps nuscenes-devkit, calls partition_by_location.
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(
    nusc_root: str = NUSC_ROOT,
    version: str = VERSION,
    out_path: str = OUT_PATH,
    train_ratio: float = TRAIN_RATIO,
) -> dict[str, list[str]]:
    """Load nuScenes metadata and emit splits.json.

    Requires nuscenes-devkit on the path. Intended to run on Kaggle.
    """
    from nuscenes.nuscenes import NuScenes  # late import — devkit only on Kaggle

    nusc = NuScenes(version=version, dataroot=nusc_root, verbose=False)
    logs_by_token = {log["token"]: log for log in nusc.log}
    splits = partition_by_location(nusc.scene, logs_by_token, train_ratio)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Offline self-test — runs without nuscenes-devkit or the dataset.
# Useful for verifying the split logic from your laptop.
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Mock 10 scenes (6 Boston, 4 Singapore) and assert split shape."""
    logs = [
        {"token": "L_B", "location": "boston-seaport"},
        {"token": "L_S1", "location": "singapore-onenorth"},
        {"token": "L_S2", "location": "singapore-queenstown"},
    ]
    logs_by_token = {l["token"]: l for l in logs}

    scenes = [
        {"token": f"sB{i}", "log_token": "L_B", "name": f"scene-010{i}"}
        for i in range(6)
    ] + [
        {"token": f"sS{i}", "log_token": "L_S1", "name": f"scene-020{i}"}
        for i in range(2)
    ] + [
        {"token": f"sS{i+2}", "log_token": "L_S2", "name": f"scene-030{i}"}
        for i in range(2)
    ]

    out = partition_by_location(scenes, logs_by_token, train_ratio=0.6)

    assert len(out["boston_train"]) + len(out["boston_val"]) == 6, out
    assert len(out["singapore_train"]) + len(out["singapore_val"]) == 4, out
    assert len(out["boston_val"]) >= 1
    assert len(out["singapore_val"]) >= 1
    assert all(isinstance(t, str) for t in out["boston_train"])

    print("self-test OK")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    if os.environ.get("LABSD_SELF_TEST") == "1":
        _self_test()
    else:
        splits = build_splits()
        print(json.dumps(splits, indent=2))
