"""Build Boston/Singapore scene-token splits from nuScenes mini."""

import json
import os
from pathlib import Path


NUSC_ROOT = "/kaggle/input/nuscenes-mini/v1.0-mini"
OUT_PATH = "/kaggle/working/splits/splits.json"


def build_splits(nusc_root: str = NUSC_ROOT, out_path: str = OUT_PATH) -> dict:
    """Partition mini scenes by location into train/val for each city."""
    # TODO: import NuScenes, iterate scenes, split by log.location
    # boston_train, boston_val, singapore_train, singapore_val
    raise NotImplementedError("placeholder")


if __name__ == "__main__":
    splits = build_splits()
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    print(json.dumps(splits, indent=2))
