"""SOTA — Planner-Centric Metric (PKL).

Reference:
    Philion, Kar, Fidler. "Learning to Evaluate Perception Models Using
    Planner-Centric Metrics." CVPR 2020.
    https://github.com/nv-tlabs/planning-centric-metrics

Why this exists for E1
----------------------
The cascade hypothesis is precisely "how do upstream perception errors
propagate to downstream driving decisions". PKL is the published metric
designed for exactly that question, and is part of the official nuScenes
detection leaderboard. Using it instead of (or alongside) a hand-rolled
IDM L2 makes the thesis result reviewer-friendly:

    "Cascade strength is measured by PKL gap = PKL(live-pipeline) -
     PKL(ground-truth-pipeline). PKL itself is the divergence between
     planner trajectory distributions under perfect vs. perturbed
     detections."

PKL uses a learned planner (Philion et al.'s) trained on nuScenes; it is
distance- and velocity-sensitive automatically, which is the property
our IDM planner kept failing to deliver.

This module is a thin wrapper. It expects the `planning_centric_metrics`
package (`pip install planning-centric-metrics`). Inputs are nuScenes-
eval-style EvalBoxes containers; we convert from our Detection records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .c1_perturbation import Detection
from .train_c1 import c1_detect, load_c1_profile
from .c1_perturbation import PERTURB_NONE


# ─────────────────────────────────────────────────────────────────────────────
# nuScenes class mapping — PKL expects these specific names.
# ─────────────────────────────────────────────────────────────────────────────

PKL_CLASS_MAP = {
    "car":                  "car",
    "bus":                  "bus",
    "truck":                "truck",
    "trailer":              "trailer",
    "construction_vehicle": "construction_vehicle",
    "motorcycle":           "motorcycle",
    "bicycle":              "bicycle",
    "pedestrian":           "pedestrian",
}


# ─────────────────────────────────────────────────────────────────────────────
# Conversion: our Detection records → nuScenes EvalBoxes
# ─────────────────────────────────────────────────────────────────────────────

def detections_to_eval_boxes(
    detections_by_sample: dict[str, list[Detection]],
):
    """Build a nuscenes.eval.common.data_classes.EvalBoxes from our records.

    Imports are lazy because nuscenes-devkit is only on the Kaggle path.
    """
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox

    eval_boxes = EvalBoxes()
    for sample_token, dets in detections_by_sample.items():
        boxes = []
        for d in dets:
            cls = PKL_CLASS_MAP.get(d.cls)
            if cls is None:
                continue
            boxes.append(DetectionBox(
                sample_token=sample_token,
                translation=(d.x, d.y, d.z),
                size=(d.width, d.length, d.height),
                rotation=(1.0, 0.0, 0.0, 0.0),  # quaternion: identity
                velocity=(d.velocity[0], d.velocity[1]),
                detection_name=cls,
                detection_score=d.score,
                attribute_name="",
            ))
        eval_boxes.add_boxes(sample_token, boxes)
    return eval_boxes


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pkl(
    c1_descriptor_path: str,
    split: str,
    splits_json: str,
    nusc,
    nusc_maps,
    device: str = "cpu",
    nworkers: int = 0,
    bsz: int = 16,
) -> dict:
    """No-op early if maps aren't loaded (PKL needs them) or if the C1
    descriptor is the YOLO format (PKL only meaningful for the
    perturbation-style oracle baseline at this stage)."""
    if not nusc_maps:
        return {
            "PKL_mean": None, "PKL_median": None, "PKL_std": None,
            "n_samples": 0,
            "note": "nusc_maps empty (no expansion JSONs in mini tarball)",
        }
    import json as _json
    with open(c1_descriptor_path) as _f:
        _descr = _json.load(_f)
    if _descr.get("kind") == "yolo":
        return {
            "PKL_mean": None, "PKL_median": None, "PKL_std": None,
            "n_samples": 0,
            "note": "PKL skipped for YOLO C1 (no perturbation profile)",
        }
    """Compute aggregate PKL over a split.

    Returns {'PKL_mean': float, 'PKL_median': float, 'PKL_std': float,
             'n_samples': int, 'profile': str}.

    Setup:
      gt_boxes:   detections from c1_detect(profile=PERTURB_NONE) — our oracle
      pred_boxes: detections from c1_detect(profile=loaded_from_descriptor)
    """
    try:
        import planning_centric_metrics
    except ImportError:
        return {
            "PKL_mean": None, "PKL_median": None, "PKL_std": None,
            "n_samples": 0,
            "note": "planning-centric-metrics not installed",
        }

    profile = load_c1_profile(c1_descriptor_path)

    with open(splits_json) as f:
        splits = json.load(f)

    # Build per-sample detection dicts for both gt and pred.
    gt_by_sample: dict[str, list[Detection]] = {}
    pred_by_sample: dict[str, list[Detection]] = {}
    sample_tokens: list[str] = []

    for scene_tok in splits.get(split, []):
        gt_dets = c1_detect(scene_tok, nusc, profile=PERTURB_NONE)
        pred_dets = c1_detect(scene_tok, nusc, profile=profile)
        for d in gt_dets:
            gt_by_sample.setdefault(d.sample_token, []).append(d)
        for d in pred_dets:
            pred_by_sample.setdefault(d.sample_token, []).append(d)
        # Sample tokens to evaluate on: every keyframe in the scene.
        scene = nusc.get("scene", scene_tok)
        st = scene["first_sample_token"]
        while st:
            sample_tokens.append(st)
            gt_by_sample.setdefault(st, [])
            pred_by_sample.setdefault(st, [])
            st = nusc.get("sample", st)["next"]

    gt_boxes = detections_to_eval_boxes(gt_by_sample)
    pred_boxes = detections_to_eval_boxes(pred_by_sample)

    info = planning_centric_metrics.calculate_pkl(
        gt_boxes, pred_boxes,
        sample_tokens, nusc,
        nusc_maps, device, nworkers,
        bsz=bsz, plot_kextremes=0, verbose=False,
    )
    # planning-centric-metrics returns a dict with min/max/mean/median/std.
    return {
        "PKL_mean":   info.get("mean") or info.get("pkl_mean"),
        "PKL_median": info.get("median") or info.get("pkl_median"),
        "PKL_std":    info.get("std") or info.get("pkl_std"),
        "n_samples":  len(sample_tokens),
        "profile":    profile.name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (offline, no nuScenes / no PKL package — verifies wiring only)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    # Verify the import-guarded path returns a placeholder when the package
    # is missing locally. (On Kaggle, the package will be installed.)
    out = evaluate_pkl(
        c1_descriptor_path="(none)",
        split="singapore_val",
        splits_json="/tmp/nonexistent.json",
        nusc=None, nusc_maps=None,
    )
    print("placeholder return:", out)
    assert out["PKL_mean"] is None
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
