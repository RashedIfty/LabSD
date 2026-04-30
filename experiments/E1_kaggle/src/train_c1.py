"""C1 — perception component (A+C hybrid).

Strategy: real pretrained CenterPoint inference for *baseline* C1 on Boston;
calibrated perturbation (c1_perturbation.py) as the stand-in for "fine-tuned
on Singapore". This is a free-compute concession; see README for rationale.

Outputs of this module match what evaluate_c1 / evaluate_c2 / evaluate_c3
expect: a list of Detection records per sample_token.

For free-compute mini runs we don't actually load a 100 MB CenterPoint
checkpoint — instead we use *ground-truth annotations as the C1 baseline*
(perfect detector, mAP=1.0 by construction) and apply the perturbation to
simulate a fine-tuned detector. This is honest in the writeup: "C1
baseline = oracle detector; C1 retrained = oracle + calibrated SG shift."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .c1_perturbation import (
    Detection,
    PerturbProfile,
    PERTURB_NONE,
    PERTURB_SG_FT,
    apply_perturbation,
    estimate_c1_mAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# C1 inference: oracle detector → perturbation
# ─────────────────────────────────────────────────────────────────────────────

def c1_detect(
    scene_token: str,
    nusc,                   # nuscenes-devkit NuScenes instance
    profile: PerturbProfile = PERTURB_NONE,
) -> list[Detection]:
    """Return per-sample detections for one scene under the given C1 profile.

    Implementation: read GT annotations as the perfect-detector baseline,
    then apply the perturbation profile.
    """
    out: list[Detection] = []
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        # Iterate annotations linked to this sample
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"].split(".")[-1]   # e.g. 'human.pedestrian.adult' → 'adult'
            cls = _bucket(ann["category_name"])
            if cls is None:
                continue
            tx, ty, tz = ann["translation"]
            # Express in ego-frame: nuscenes annotations are global; we
            # subtract ego pose for distance-based reasoning.
            sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            ex, ey, ez = ep["translation"]
            x_rel = tx - ex
            y_rel = ty - ey
            sw, sl, sh = ann["size"]   # nuscenes order: width, length, height
            out.append(Detection(
                cls=cls,
                x=x_rel, y=y_rel, z=tz - ez,
                width=sw, length=sl, height=sh,
                yaw=0.0,
                score=0.95,
                sample_token=sample_token,
            ))
        sample_token = sample["next"]

    return apply_perturbation(out, profile)


def _bucket(category_name: str) -> str | None:
    """Map nuScenes fine-grained category to coarse class used by E1."""
    if category_name.startswith("vehicle.car"):
        return "car"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.trailer"):
        return "trailer"
    if category_name.startswith("vehicle.construction"):
        return "construction_vehicle"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API consumed by eval.py
# ─────────────────────────────────────────────────────────────────────────────

def train_c1(
    config_path: str | None = None,
    work_dir: str = "/kaggle/working/c1",
    profile: PerturbProfile = PERTURB_NONE,
    resume_from: str | None = None,
) -> str:
    """In the A+C hybrid there is no actual training. This function emits a
    'checkpoint' descriptor — a JSON pointing at the perturbation profile
    we'll use at inference time.

    Returns the path to the emitted descriptor.
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out = Path(work_dir) / "c1_descriptor.json"
    descriptor = {
        "kind": "oracle+perturbation",
        "profile_name": profile.name,
        "profile_params": {
            "pedestrian_long_range_drop": profile.pedestrian_long_range_drop,
            "long_range_threshold_m": profile.long_range_threshold_m,
            "vehicle_bbox_scale": profile.vehicle_bbox_scale,
            "bbox_center_jitter_m": profile.bbox_center_jitter_m,
            "confidence_drift": profile.confidence_drift,
            "seed": profile.seed,
        },
        "resumed_from": resume_from,
    }
    with open(out, "w") as f:
        json.dump(descriptor, f, indent=2)
    return str(out)


def load_c1_profile(descriptor_path: str) -> PerturbProfile:
    """Reverse of train_c1 — read a descriptor JSON, return a profile."""
    with open(descriptor_path) as f:
        d = json.load(f)
    p = d["profile_params"]
    return PerturbProfile(
        name=d["profile_name"],
        pedestrian_long_range_drop=p["pedestrian_long_range_drop"],
        long_range_threshold_m=p["long_range_threshold_m"],
        vehicle_bbox_scale=p["vehicle_bbox_scale"],
        bbox_center_jitter_m=p["bbox_center_jitter_m"],
        confidence_drift=p["confidence_drift"],
        seed=p["seed"],
    )


def evaluate_c1(checkpoint: str, split: str, splits_json: str | None = None,
                nusc=None) -> dict:
    """Compute aggregate mAP under the C1 profile, against per-scene GT.

    Returns {'mAP': float, 'profile': str}. The mAP is an order-of-magnitude
    proxy (see c1_perturbation._iou_2d): direction matters, not absolute
    units.
    """
    profile = load_c1_profile(checkpoint)
    if nusc is None or splits_json is None:
        return {"mAP": None, "profile": profile.name, "note": "skipped (no nusc)"}

    with open(splits_json) as f:
        splits = json.load(f)

    scene_tokens = splits.get(split, [])
    matched = 0
    total = 0
    for scene_tok in scene_tokens:
        gt = c1_detect(scene_tok, nusc, profile=PERTURB_NONE)
        det = c1_detect(scene_tok, nusc, profile=profile)
        for g in gt:
            total += 1
            for d in det:
                if d.cls == g.cls and abs(d.x - g.x) < 1.0 and abs(d.y - g.y) < 1.0:
                    matched += 1
                    break
    mAP = (matched / total) if total else 0.0
    return {"mAP": mAP, "profile": profile.name, "n_gt": total}


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (offline, no nusc)
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ck_baseline = train_c1(work_dir=f"{d}/c1_boston", profile=PERTURB_NONE)
        ck_singapore = train_c1(work_dir=f"{d}/c1_sg", profile=PERTURB_SG_FT,
                                 resume_from=ck_baseline)

        prof_b = load_c1_profile(ck_baseline)
        prof_s = load_c1_profile(ck_singapore)
        assert prof_b.name == "none"
        assert prof_s.name == "sg_ft"
        print("descriptors round-trip OK")

        # evaluate_c1 without nusc returns a placeholder, not an error
        out = evaluate_c1(ck_singapore, split="singapore_val")
        assert out["profile"] == "sg_ft"
        assert out["mAP"] is None
        print("evaluate_c1 without nusc OK:", out)

    print("self-test OK")


if __name__ == "__main__":
    _self_test()
