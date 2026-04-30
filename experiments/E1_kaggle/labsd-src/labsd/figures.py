"""Presentation-grade figures for E1.

Generated post-run from baseline.json + after_retrain.json + the YOLO
training output dirs. Every plot is saved at 150 DPI to /kaggle/working/figs/.

Tier-1 figures:
    - fig01_table_full.png            full Table I bar chart, all 5 metrics
    - fig02_cascade_planning.png      C3 isolated vs pipeline before/after
                                      (already exists; re-emit with deltas)
    - fig03_pipeline_diagram.png      annotated C1 → C2 → C3 with numbers
    - fig04_yolo_curves_boston.png    Ultralytics results.png copy
    - fig05_yolo_curves_singapore.png Ultralytics results.png copy
    - fig06_confusion_boston.png      Ultralytics confusion_matrix.png copy
    - fig07_confusion_singapore.png   Ultralytics confusion_matrix.png copy
    - fig08_detection_grid.png        side-by-side Boston-YOLO vs SG-YOLO
                                      detections on shared val images

Tier-2 figures:
    - fig09_per_scene_l2.png          per-scene C3 pipeline L2 before/after
    - fig10_failure_modes.png         worst-delta scenes annotated
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable


FIG_DIR = "/kaggle/working/figs"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(p: str = FIG_DIR) -> Path:
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def _bar_with_deltas(ax, labels, before, after, ylabel, title):
    import numpy as np
    x = np.arange(len(labels))
    w = 0.38
    b = ax.bar(x - w / 2, before, w, label="Before (Boston-trained C1)",
               color="#3b82f6")
    a = ax.bar(x + w / 2, after, w, label="After (SG-fine-tuned C1)",
               color="#f97316")
    for rect, val in zip(b, before):
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            ax.text(rect.get_x() + rect.get_width()/2, val, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)
    for rect, val in zip(a, after):
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            ax.text(rect.get_x() + rect.get_width()/2, val, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1
# ─────────────────────────────────────────────────────────────────────────────

def fig01_full_table(baseline_json: str, after_json: str, out_dir: str = FIG_DIR):
    """All five Table I metrics on one chart. Two-subplot layout because mAP
    is on a 0-1 scale while minADE/L2 are in metres."""
    import matplotlib.pyplot as plt
    out = _ensure_dir(out_dir)
    before = json.load(open(baseline_json))
    after = json.load(open(after_json))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mAP (0-1 scale, single bar per condition)
    mAP_b = before.get("c1_mAP")
    mAP_a = after.get("c1_mAP")
    _bar_with_deltas(
        ax1, ["C1 mAP@50"], [mAP_b], [mAP_a],
        ylabel="mAP@50 (0-1)",
        title="C1 detection quality on Singapore val",
    )

    # Right: distance-based metrics
    labels = [
        "C2 isolated\nminADE",
        "C2 pipeline\nminADE",
        "C3 isolated\nL2@3s",
        "C3 pipeline\nL2@3s",
    ]
    bef = [
        before.get("c2_iso_minADE"),
        before.get("c2_pipe_minADE"),
        before.get("c3_iso_L2"),
        before.get("c3_pipe_L2"),
    ]
    aft = [
        after.get("c2_iso_minADE"),
        after.get("c2_pipe_minADE"),
        after.get("c3_iso_L2"),
        after.get("c3_pipe_L2"),
    ]
    _bar_with_deltas(
        ax2, labels, bef, aft,
        ylabel="error (m)",
        title="C2 + C3 prediction / planning error",
    )
    fig.suptitle("E1 — Table I: full pipeline, before vs after C1 retrain",
                 fontsize=13)
    fig.tight_layout()
    p = out / "fig01_table_full.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def fig02_cascade_planning(baseline_json: str, after_json: str,
                           out_dir: str = FIG_DIR):
    """Headline C3-isolated vs C3-pipeline figure with explicit deltas."""
    import matplotlib.pyplot as plt
    out = _ensure_dir(out_dir)
    before = json.load(open(baseline_json))
    after = json.load(open(after_json))

    fig, ax = plt.subplots(figsize=(8, 5))
    _bar_with_deltas(
        ax,
        ["C3 (isolated)\n[GT inputs]", "C3 (pipeline)\n[live inputs]"],
        [before["c3_iso_L2"],  before["c3_pipe_L2"]],
        [after["c3_iso_L2"],   after["c3_pipe_L2"]],
        ylabel="L2 error vs human driver @ 3s (m)",
        title="Cascade signature: planning under retrained C1",
    )
    # Annotate the delta on C3 pipeline
    delta = after["c3_pipe_L2"] - before["c3_pipe_L2"]
    pct = 100 * delta / before["c3_pipe_L2"] if before["c3_pipe_L2"] else 0
    ax.annotate(
        f"Δ = {delta:+.3f} m  ({pct:+.2f}%)",
        xy=(1, max(before["c3_pipe_L2"], after["c3_pipe_L2"])),
        xytext=(0.5, max(before["c3_pipe_L2"], after["c3_pipe_L2"]) + 0.5),
        fontsize=11, ha="center",
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
        color="#dc2626",
    )
    fig.tight_layout()
    p = out / "fig02_cascade_planning.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def fig03_pipeline_diagram(baseline_json: str, after_json: str,
                           out_dir: str = FIG_DIR):
    """C1 → C2 → C3 boxes with measured before/after numbers on each arrow."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    out = _ensure_dir(out_dir)
    before = json.load(open(baseline_json))
    after = json.load(open(after_json))

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (1.0, "C1\nYOLOv11", "#3b82f6"),
        (5.0, "C2\nconst-velocity", "#8b5cf6"),
        (9.0, "C3\nIDM planner", "#10b981"),
    ]
    for x, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 1.5), 2, 1.2,
            boxstyle="round,pad=0.08",
            linewidth=2, edgecolor=color, facecolor="white",
        ))
        ax.text(x + 1, 2.1, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

    # Arrows
    for x in (3.0, 7.0):
        ax.annotate("", xy=(x + 2, 2.1), xytext=(x, 2.1),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#374151"))

    # Metric annotations
    def fmt(v): return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"
    c1_mAP_b = before.get("c1_mAP"); c1_mAP_a = after.get("c1_mAP")
    ax.text(2.0, 0.9, f"C1 mAP@50\nbefore: {fmt(c1_mAP_b)}\nafter:  {fmt(c1_mAP_a)}",
            ha="center", va="top", fontsize=9, color="#1f2937")
    ax.text(6.0, 0.9, f"C2 minADE (m)\niso  {fmt(before['c2_iso_minADE'])} → {fmt(after['c2_iso_minADE'])}\npipe {fmt(before['c2_pipe_minADE'])} → {fmt(after['c2_pipe_minADE'])}",
            ha="center", va="top", fontsize=9, color="#1f2937")
    ax.text(10.0, 0.9, f"C3 L2@3s (m)\niso  {fmt(before['c3_iso_L2'])} → {fmt(after['c3_iso_L2'])}\npipe {fmt(before['c3_pipe_L2'])} → {fmt(after['c3_pipe_L2'])}",
            ha="center", va="top", fontsize=9, color="#1f2937")

    ax.text(6.0, 3.6, "E1 pipeline (Boston-trained C1 → Singapore-fine-tuned C1)",
            ha="center", va="center", fontsize=12, fontweight="bold")

    p = out / "fig03_pipeline_diagram.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def copy_yolo_artifacts(yolo_run_dir: str, prefix: str, out_dir: str = FIG_DIR):
    """Copy Ultralytics-generated training plots into the figures dir.

    Ultralytics writes:
        results.png            -- loss + mAP curves
        confusion_matrix.png   -- normalized confusion matrix
        labels.jpg             -- label distribution
        train_batch0.jpg       -- annotated training batch
        val_batch0_pred.jpg    -- val batch predictions
    """
    out = _ensure_dir(out_dir)
    src = Path(yolo_run_dir)
    found = []
    name_map = {
        "results.png":          f"{prefix}_curves.png",
        "confusion_matrix.png": f"{prefix}_confusion.png",
        "labels.jpg":           f"{prefix}_labels.jpg",
        "val_batch0_pred.jpg":  f"{prefix}_val_pred.jpg",
        "val_batch0_labels.jpg":f"{prefix}_val_labels.jpg",
    }
    for src_name, dst_name in name_map.items():
        s = src / src_name
        if s.exists():
            d = out / dst_name
            shutil.copy2(s, d)
            found.append(str(d))
    return found


def fig08_detection_grid(
    nusc,
    splits_json: str,
    boston_weights: str,
    singapore_weights: str,
    out_dir: str = FIG_DIR,
    n_samples: int = 3,
    cam_channel: str = "CAM_FRONT",
):
    """Side-by-side: same images run through Boston-YOLO vs SG-YOLO."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO

    out = _ensure_dir(out_dir)
    splits = json.load(open(splits_json))
    sg_val = splits.get("singapore_val", [])

    # Pick the first n_samples keyframes from sg_val.
    img_paths = []
    for scene_tok in sg_val:
        scene = nusc.get("scene", scene_tok)
        sample_token = scene["first_sample_token"]
        # Sample every ~5th frame to get a spread
        i = 0
        while sample_token and len(img_paths) < n_samples * len(sg_val):
            sample = nusc.get("sample", sample_token)
            cam_data_tok = sample["data"].get(cam_channel)
            if cam_data_tok and i % 5 == 0:
                cam_data = nusc.get("sample_data", cam_data_tok)
                img_paths.append(os.path.join(nusc.dataroot, cam_data["filename"]))
            sample_token = sample["next"]
            i += 1
            if len(img_paths) >= n_samples:
                break
        if len(img_paths) >= n_samples:
            break

    if not img_paths:
        print("[figures] no Singapore val images found")
        return None

    boston_model = YOLO(boston_weights)
    sg_model = YOLO(singapore_weights)

    fig, axes = plt.subplots(len(img_paths), 2, figsize=(14, 4 * len(img_paths)))
    if len(img_paths) == 1:
        axes = [axes]

    for row, img_path in enumerate(img_paths):
        for col, (model, label) in enumerate([
            (boston_model, "Boston-trained C1"),
            (sg_model, "Singapore-fine-tuned C1"),
        ]):
            results = model(img_path, conf=0.25, verbose=False)
            annotated = results[0].plot()
            axes[row][col].imshow(annotated[..., ::-1])  # BGR → RGB
            axes[row][col].set_title(
                f"{label}\n{Path(img_path).name[:40]}",
                fontsize=10,
            )
            axes[row][col].axis("off")

    fig.suptitle("YOLOv11 detections on Singapore val "
                 "(same images, before vs after fine-tune)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    p = out / "fig08_detection_grid.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2
# ─────────────────────────────────────────────────────────────────────────────

def fig09_per_scene_l2(baseline_json: str, after_json: str,
                       out_dir: str = FIG_DIR):
    """Per-scene C3 pipeline L2 — shows which scenes drive the +4.45%."""
    import matplotlib.pyplot as plt
    import numpy as np
    out = _ensure_dir(out_dir)
    before = json.load(open(baseline_json))
    after = json.load(open(after_json))
    bp = before.get("c3_pipe_per_scene") or []
    ap = after.get("c3_pipe_per_scene") or []
    if not bp or not ap:
        print("[figures] per_scene data missing; skip fig09")
        return None
    n = min(len(bp), len(ap))
    bp, ap = bp[:n], ap[:n]
    labels = [s["scene_token"][:8] for s in bp]
    b_l2 = [s.get("L2@3s") or 0 for s in bp]
    a_l2 = [s.get("L2@3s") or 0 for s in ap]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    x = np.arange(n)
    w = 0.4
    ax.bar(x - w / 2, b_l2, w, label="Boston-trained C1", color="#3b82f6")
    ax.bar(x + w / 2, a_l2, w, label="SG-fine-tuned C1", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("C3 pipeline L2@3s (m)")
    ax.set_title("Per-scene C3 pipeline L2 — Singapore val")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "fig09_per_scene_l2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def fig10_failure_modes(baseline_json: str, after_json: str,
                        out_dir: str = FIG_DIR):
    """Per-scene delta — highlights worst regressions."""
    import matplotlib.pyplot as plt
    import numpy as np
    out = _ensure_dir(out_dir)
    before = json.load(open(baseline_json))
    after = json.load(open(after_json))
    bp = before.get("c3_pipe_per_scene") or []
    ap = after.get("c3_pipe_per_scene") or []
    if not bp or not ap:
        return None
    n = min(len(bp), len(ap))
    deltas = [(ap[i].get("L2@3s") or 0) - (bp[i].get("L2@3s") or 0)
              for i in range(n)]
    labels = [bp[i]["scene_token"][:8] for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    colors = ["#dc2626" if d > 0 else "#16a34a" for d in deltas]
    ax.bar(np.arange(n), deltas, color=colors)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Δ C3 pipeline L2@3s (m)\n[after − before]")
    ax.set_title("Per-scene cascade contribution: "
                 "red = retrain hurt planning, green = retrain helped")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "fig10_failure_modes.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level driver
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(
    baseline_json: str,
    after_json: str,
    yolo_boston_dir: str,
    yolo_singapore_dir: str,
    boston_weights: str,
    singapore_weights: str,
    nusc=None,
    splits_json: str | None = None,
    out_dir: str = FIG_DIR,
) -> list[str]:
    """Produce the full presentation set. Returns list of figure paths."""
    paths: list[str] = []
    paths.append(fig01_full_table(baseline_json, after_json, out_dir))
    paths.append(fig02_cascade_planning(baseline_json, after_json, out_dir))
    paths.append(fig03_pipeline_diagram(baseline_json, after_json, out_dir))
    paths += copy_yolo_artifacts(yolo_boston_dir, "fig04_boston", out_dir)
    paths += copy_yolo_artifacts(yolo_singapore_dir, "fig05_singapore", out_dir)
    if nusc is not None and splits_json is not None:
        p = fig08_detection_grid(
            nusc, splits_json, boston_weights, singapore_weights, out_dir,
        )
        if p: paths.append(p)
    p9 = fig09_per_scene_l2(baseline_json, after_json, out_dir)
    if p9: paths.append(p9)
    p10 = fig10_failure_modes(baseline_json, after_json, out_dir)
    if p10: paths.append(p10)
    return [p for p in paths if p]


def _self_test() -> None:
    """No-arg sanity check the module imports cleanly."""
    import inspect
    assert callable(generate_all)
    assert callable(fig01_full_table)
    assert callable(copy_yolo_artifacts)
    print("figures module OK")


if __name__ == "__main__":
    _self_test()
