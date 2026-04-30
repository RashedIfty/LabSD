"""C2 — prediction component (A+C hybrid).

Strategy: deterministic constant-velocity-per-agent rollouts as the
prediction model. No training (mini has only 2 Boston train scenes —
MTP would overfit instantly). The cascade signal we measure does not
depend on prediction-model sophistication; it depends on whether C2
receives consistent inputs across the experiment.

Two evaluation modes (matches Meeting 3 §VI Table I):
  - isolated: predictions made from ground-truth agent positions/velocities
  - pipeline: predictions made from C1's outputs (which include the
              calibrated SG perturbation when "C1 retrained" is active)

For each agent, predict its future position over a 3-second horizon at
0.5 s steps using its current velocity (from nuScenes annotations
in isolated mode, or zero-velocity guess from a single C1 frame in
pipeline mode — this is exactly the kind of degradation the cascade
hypothesis predicts).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .c1_perturbation import Detection
from .train_c1 import c1_detect, load_c1_profile


HORIZON_S = 3.0
DT = 0.5
N_STEPS = int(round(HORIZON_S / DT))


@dataclass
class AgentPrediction:
    sample_token: str
    cls: str
    waypoints: list[tuple[float, float]] = field(default_factory=list)  # (x, y) per step
    initial_pos: tuple[float, float] = (0.0, 0.0)
    initial_vel: tuple[float, float] = (0.0, 0.0)
    instance_token: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Public API consumed by eval.py
# ─────────────────────────────────────────────────────────────────────────────

def train_c2(
    split: str = "boston_train",
    out_path: str = "/kaggle/working/c2.json",
) -> str:
    """No real training — emits a descriptor JSON identifying the C2 model.

    Mirrors train_c1's approach: the heavy lifting is in c2_predict at
    inference time; train_c2 just records that 'this checkpoint = constant
    velocity model'.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "kind": "constant_velocity",
        "horizon_s": HORIZON_S,
        "dt": DT,
        "trained_on": split,
    }
    with open(out_path, "w") as f:
        json.dump(descriptor, f, indent=2)
    return out_path


def c2_predict(
    scene_token: str,
    nusc,
    mode: str,                              # 'isolated' | 'pipeline'
    c1_descriptor_path: str | None = None,
) -> list[AgentPrediction]:
    """Return per-agent trajectories for one scene.

    isolated: read GT instance trajectories — each agent's velocity is
              computed from consecutive annotation frames.
    pipeline: read C1 detections (single-frame), velocity unknown so we
              fall back to zero-velocity rollouts. This is the part that
              degrades when C1 is perturbed.
    """
    if mode == "isolated":
        return _predict_from_gt(scene_token, nusc)
    elif mode == "pipeline":
        from .c1_perturbation import PERTURB_NONE
        profile = (
            load_c1_profile(c1_descriptor_path)
            if c1_descriptor_path
            else PERTURB_NONE
        )
        return _predict_from_detections(scene_token, nusc, profile)
    else:
        raise ValueError(f"unknown mode: {mode}")


def _predict_from_gt(scene_token: str, nusc) -> list[AgentPrediction]:
    """Use consecutive nuScenes annotations to estimate velocity, then
    constant-velocity roll out from the first sample.

    All positions are in heading-aligned ego-frame of the *first* sample,
    so they're directly comparable to planner output.
    """
    scene = nusc.get("scene", scene_token)

    # Anchor frame: first sample's ego pose + heading.
    sample0 = nusc.get("sample", scene["first_sample_token"])
    sd0 = nusc.get("sample_data", sample0["data"]["LIDAR_TOP"])
    ep0 = nusc.get("ego_pose", sd0["ego_pose_token"])
    ax, ay, _ = ep0["translation"]
    qw, qx, qy, qz = ep0["rotation"]
    heading0 = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cos_h, sin_h = math.cos(-heading0), math.sin(-heading0)

    # Build instance → list[(sample_token, t, ego_x, ego_y)] timeline.
    instance_traj: dict[str, list[tuple[str, float, float, float]]] = {}
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        t = sample["timestamp"] / 1e6
        for ann_tok in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            inst = ann["instance_token"]
            tx, ty, _ = ann["translation"]
            dx, dy = tx - ax, ty - ay
            ego_x = cos_h * dx - sin_h * dy
            ego_y = sin_h * dx + cos_h * dy
            instance_traj.setdefault(inst, []).append((sample_token, t, ego_x, ego_y))
        sample_token = sample["next"]

    out: list[AgentPrediction] = []
    for inst_tok, frames in instance_traj.items():
        if len(frames) < 2:
            continue
        first_sample, t0, x0, y0 = frames[0]
        _, t1, x1, y1 = frames[1]
        dt = t1 - t0
        if dt <= 0:
            continue
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
        # Get class from instance
        ann = nusc.get("sample_annotation", nusc.get("instance", inst_tok)["first_annotation_token"])
        cls = _bucket(ann["category_name"])
        if cls is None:
            continue
        wpts = [
            (x0 + vx * (i * DT), y0 + vy * (i * DT))
            for i in range(1, N_STEPS + 1)
        ]
        out.append(AgentPrediction(
            sample_token=first_sample,
            cls=cls,
            waypoints=wpts,
            initial_pos=(x0, y0),
            initial_vel=(vx, vy),
            instance_token=inst_tok,
        ))
    return out


def _actual_future_positions(scene_token: str, nusc) -> dict[str, list[tuple[float, float]]]:
    """Per-instance list of (ego_x, ego_y) at sample frames *after* t=0,
    in heading-aligned ego-frame of the first sample. The constant-velocity
    isolated-mode prediction is compared against this.
    """
    scene = nusc.get("scene", scene_token)
    sample0 = nusc.get("sample", scene["first_sample_token"])
    sd0 = nusc.get("sample_data", sample0["data"]["LIDAR_TOP"])
    ep0 = nusc.get("ego_pose", sd0["ego_pose_token"])
    ax, ay, _ = ep0["translation"]
    qw, qx, qy, qz = ep0["rotation"]
    heading0 = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cos_h, sin_h = math.cos(-heading0), math.sin(-heading0)

    futures: dict[str, list[tuple[float, float]]] = {}
    sample_token = sample0["next"]   # skip t=0; start from t=dt
    while sample_token:
        sample = nusc.get("sample", sample_token)
        for ann_tok in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            inst = ann["instance_token"]
            tx, ty, _ = ann["translation"]
            dx, dy = tx - ax, ty - ay
            ego_x = cos_h * dx - sin_h * dy
            ego_y = sin_h * dx + cos_h * dy
            futures.setdefault(inst, []).append((ego_x, ego_y))
        sample_token = sample["next"]
    return futures


def _inst_for_agent(ap, scene_token, nusc) -> str:
    """Compatibility shim — preferred is ap.instance_token; fall back to None."""
    return ap.instance_token or ""


def _predict_from_detections(scene_token: str, nusc, profile) -> list[AgentPrediction]:
    """Single-frame detections → zero-velocity prediction.

    This deliberately models the failure mode: with no temporal history
    (just one frame from C1), C2 cannot estimate velocity and falls back
    to assuming agents are stationary. This is the cascade pathway.
    """
    dets = c1_detect(scene_token, nusc, profile=profile)
    # Group by sample_token; only use the first sample (head of scene)
    if not dets:
        return []
    first_sample = dets[0].sample_token
    out: list[AgentPrediction] = []
    for d in dets:
        if d.sample_token != first_sample:
            continue
        # Zero-velocity rollout: agent stays put.
        wpts = [(d.x, d.y) for _ in range(1, N_STEPS + 1)]
        out.append(AgentPrediction(
            sample_token=d.sample_token,
            cls=d.cls,
            waypoints=wpts,
            initial_pos=(d.x, d.y),
            initial_vel=(0.0, 0.0),
        ))
    return out


def _bucket(category_name: str) -> str | None:
    if category_name.startswith("vehicle.car"): return "car"
    if category_name.startswith("vehicle.bus"): return "bus"
    if category_name.startswith("vehicle.truck"): return "truck"
    if category_name.startswith("vehicle.trailer"): return "trailer"
    if category_name.startswith("vehicle.construction"): return "construction_vehicle"
    if category_name.startswith("vehicle.motorcycle"): return "motorcycle"
    if category_name.startswith("vehicle.bicycle"): return "bicycle"
    if category_name.startswith("human.pedestrian"): return "pedestrian"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: minADE against GT future positions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_c2(
    checkpoint: str,
    split: str,
    mode: str,
    splits_json: str | None = None,
    nusc=None,
    c1_descriptor_path: str | None = None,
) -> dict:
    """minADE over a split. Returns {'minADE_5': float, 'mode': str, 'n_agents': int}.

    minADE_5 here is just minADE (no multi-modal sampling — our C2 outputs
    a single trajectory). The metric name is kept for table.py compatibility.
    """
    if nusc is None or splits_json is None:
        return {"minADE_5": None, "mode": mode, "note": "skipped (no nusc)"}

    with open(splits_json) as f:
        splits = json.load(f)

    total_err = 0.0
    n = 0
    for scene_tok in splits.get(split, []):
        gt = _predict_from_gt(scene_tok, nusc)
        actual_future = _actual_future_positions(scene_tok, nusc)
        if mode == "isolated":
            # ADE: constant-vel prediction (from gt) vs actual annotated future
            # for the same instance.
            for ap in gt:
                actual = actual_future.get(_inst_for_agent(ap, scene_tok, nusc))
                if actual is None or len(actual) == 0:
                    continue
                k = min(len(ap.waypoints), len(actual))
                if k == 0:
                    continue
                err = sum(math.hypot(ap.waypoints[i][0] - actual[i][0],
                                     ap.waypoints[i][1] - actual[i][1])
                          for i in range(k)) / k
                total_err += err
                n += 1
        elif mode == "pipeline":
            pred = _predict_from_detections(
                scene_tok, nusc,
                load_c1_profile(c1_descriptor_path)
                if c1_descriptor_path else __import__("labsd.c1_perturbation",
                                                     fromlist=["PERTURB_NONE"]).PERTURB_NONE,
            )
            # Match each prediction to closest GT initial position
            for p in pred:
                best = None
                best_d = 1e9
                for g in gt:
                    d = math.hypot(p.initial_pos[0] - g.initial_pos[0],
                                   p.initial_pos[1] - g.initial_pos[1])
                    if d < best_d and p.cls == g.cls:
                        best_d = d
                        best = g
                if best is None or best_d > 5.0:
                    continue
                # ADE between p.waypoints and best.waypoints
                k = min(len(p.waypoints), len(best.waypoints))
                if k == 0:
                    continue
                err = sum(math.hypot(p.waypoints[i][0] - best.waypoints[i][0],
                                     p.waypoints[i][1] - best.waypoints[i][1])
                          for i in range(k)) / k
                total_err += err
                n += 1

    return {
        "minADE_5": (total_err / n) if n else 0.0,
        "mode": mode,
        "n_agents": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ck = train_c2(split="boston_train", out_path=f"{d}/c2.json")
        with open(ck) as f:
            descr = json.load(f)
        assert descr["kind"] == "constant_velocity"
        assert descr["horizon_s"] == HORIZON_S
        print("descriptor OK:", descr)

        # evaluate_c2 without nusc returns placeholder
        out = evaluate_c2(ck, split="singapore_val", mode="isolated")
        assert out["mode"] == "isolated"
        assert out["minADE_5"] is None
        print("evaluate_c2 without nusc OK:", out)

    print("self-test OK")


if __name__ == "__main__":
    _self_test()
