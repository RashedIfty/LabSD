"""Class-balanced fine-tuning for C1 (Perception).

Motivation (Meeting-4 "What comes next", step 2): across the whole campaign the
Singapore fine-tune *lost* aggregate mAP ($\\delta_1 < 0$), because the 119-image
fine-tune set is small and its class mix is dominated by `car` while most other
driving classes are rare. That drop is what keeps us in the *metric-decoupling*
regime instead of the strict entangled-enhancement condition, which needs
$\\delta_1 > 0$ (C1 improves on its own metric) while the system still degrades.

This module builds a class-balanced version of a YOLO training split by
**image-level oversampling**: images containing rare classes are duplicated so
the per-class instance distribution flattens. Ultralytics does not expose
per-class loss weights in a stable way, so oversampling is the robust route and
it needs no change to the training loop.

The val split is never touched, so mAP stays comparable to every other run in
the campaign.
"""
from __future__ import annotations

import math
import shutil
from collections import Counter
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# label inspection
# ─────────────────────────────────────────────────────────────────────────────

def class_counts(dataset_dir: str, split: str = "train") -> Counter:
    """Per-class instance counts over a YOLO split's label files."""
    counts: Counter = Counter()
    labels = Path(dataset_dir) / "labels" / split
    if not labels.is_dir():
        return counts
    for p in labels.glob("*.txt"):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                counts[int(line.split()[0])] += 1
            except (ValueError, IndexError):
                continue
    return counts


def _image_for(labels_path: Path, images_dir: Path) -> Path | None:
    """Locate the image matching a label file (jpg, then png)."""
    for ext in (".jpg", ".jpeg", ".png"):
        cand = images_dir / (labels_path.stem + ext)
        if cand.exists():
            return cand
    return None


# ─────────────────────────────────────────────────────────────────────────────
# the balanced build
# ─────────────────────────────────────────────────────────────────────────────

def build_class_balanced_split(
    src_dir: str,
    dst_dir: str,
    split: str = "train",
    max_repeat: int = 12,
    power: float = 0.5,
) -> dict:
    """Write a class-balanced copy of ``src_dir``'s split into ``dst_dir``.

    Each source image is emitted once, then repeated up to ``max_repeat`` times
    according to the rarest class it contains. The repeat factor for a class is
    ``(max_count / count) ** power`` clipped to ``[1, max_repeat]``; ``power``
    softens the correction (0 = no balancing, 1 = full inverse-frequency).

    Note on what oversampling can and cannot do here: driving scenes are
    multi-label, and the dominant class (`car`) co-occurs with almost every rare
    class, so duplicating a rare-class image also duplicates its cars. The
    imbalance therefore shrinks substantially but cannot be driven to 1.0 by
    image-level repetition alone. ``max_repeat=12`` is the point past which
    further repetition mostly inflates the dataset (and training time) rather
    than flattening the mix.

    Returns a summary dict with before/after class counts and image totals.
    """
    src_images = Path(src_dir) / "images" / split
    src_labels = Path(src_dir) / "labels" / split
    dst_images = Path(dst_dir) / "images" / split
    dst_labels = Path(dst_dir) / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    before = class_counts(src_dir, split)
    if not before:
        return {"error": "no labels found", "src": src_dir, "split": split}

    max_count = max(before.values())

    # repeat factor per class
    rep_for_class: dict[int, int] = {}
    for cls, n in before.items():
        if n <= 0:
            continue
        factor = (max_count / n) ** power
        rep_for_class[cls] = int(max(1, min(max_repeat, round(factor))))

    n_src = 0
    n_written = 0
    for lp in sorted(src_labels.glob("*.txt")):
        img = _image_for(lp, src_images)
        if img is None:
            continue
        n_src += 1
        classes = set()
        for line in lp.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    classes.add(int(line.split()[0]))
                except (ValueError, IndexError):
                    pass
        # an image is repeated as often as its *rarest* class demands
        reps = max((rep_for_class.get(c, 1) for c in classes), default=1)
        for r in range(reps):
            stem = lp.stem if r == 0 else f"{lp.stem}__rep{r}"
            shutil.copy2(img, dst_images / f"{stem}{img.suffix}")
            shutil.copy2(lp, dst_labels / f"{stem}.txt")
            n_written += 1

    after = class_counts(dst_dir, split)
    return {
        "src_images": n_src,
        "balanced_images": n_written,
        "repeat_factors": {int(k): int(v) for k, v in sorted(rep_for_class.items())},
        "counts_before": {int(k): int(v) for k, v in sorted(before.items())},
        "counts_after": {int(k): int(v) for k, v in sorted(after.items())},
        "imbalance_before": _imbalance(before),
        "imbalance_after": _imbalance(after),
    }


def _imbalance(counts: Counter) -> float | None:
    """max/min instance-count ratio; 1.0 = perfectly flat."""
    if not counts:
        return None
    lo = min(counts.values())
    hi = max(counts.values())
    return hi / lo if lo > 0 else None


def copy_split(src_dir: str, dst_dir: str, split: str) -> int:
    """Copy a split verbatim (used for val, which must stay untouched)."""
    n = 0
    for kind in ("images", "labels"):
        s = Path(src_dir) / kind / split
        d = Path(dst_dir) / kind / split
        d.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            for p in s.iterdir():
                shutil.copy2(p, d / p.name)
                if kind == "images":
                    n += 1
    return n
