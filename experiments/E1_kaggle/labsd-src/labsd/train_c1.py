"""C1 — CenterPoint training wrapper."""

import subprocess
from pathlib import Path


def train_c1(
    config_path: str,
    work_dir: str,
    epochs: int = 5,
    resume_from: str | None = None,
) -> str:
    """Train (or fine-tune) CenterPoint, return path to final checkpoint."""
    # TODO: invoke `python -m mmdet3d.tools.train` with cfg-options
    # TODO: handle resume_from for Singapore fine-tune
    raise NotImplementedError("placeholder")


def evaluate_c1(checkpoint: str, split: str) -> dict:
    """Run CenterPoint eval, return {'mAP': ..., 'NDS': ...}."""
    # TODO: invoke mmdet3d test, parse output
    raise NotImplementedError("placeholder")


if __name__ == "__main__":
    ckpt = train_c1(
        config_path="configs/centerpoint_boston.py",
        work_dir="/kaggle/working/c1_boston",
    )
    print(f"saved: {ckpt}")
