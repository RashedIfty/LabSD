"""C2 — MTP (Multiple Trajectory Prediction) training wrapper."""

from pathlib import Path


def train_c2(
    split: str,
    out_path: str,
    epochs: int = 10,
) -> str:
    """Train MTP on nuScenes prediction challenge format, return checkpoint path."""
    # TODO: load ResNetBackbone + MTP from nuscenes-devkit
    # TODO: standard PyTorch training loop, save best checkpoint
    raise NotImplementedError("placeholder")


def evaluate_c2(checkpoint: str, split: str, mode: str) -> dict:
    """
    Evaluate MTP and return {'minADE_5': ..., 'minFDE_5': ...}.

    mode='isolated': feed ground-truth detections/tracks
    mode='pipeline': feed C1's detections through a tracker
    """
    # TODO: branch on mode
    raise NotImplementedError("placeholder")


if __name__ == "__main__":
    ckpt = train_c2(split="boston_train", out_path="/kaggle/working/c2_boston.pth")
    print(f"saved: {ckpt}")
