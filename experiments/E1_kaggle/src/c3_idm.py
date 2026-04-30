"""C3 — Rule-based IDM planner (PDM-Closed substitute for free-tier compute).

Captures the cascade mechanism without nuPlan dependencies. Given an ego
state and a list of predicted other agents, scores a small grid of candidate
trajectories and returns the best.

The scoring follows the spirit of PDM-Closed:
    score = w_collision * collision_penalty
          + w_progress  * progress_reward
          + w_comfort   * comfort_penalty
          + w_offroad   * offroad_penalty

For E1 on free compute we keep the planner deliberately simple: lateral
offset × target speed grid, constant-velocity rollout, axis-aligned
bounding-box collision check. The cascade signal we want to measure does
not require a sophisticated planner — it requires an *interpretable* one
whose output is a deterministic function of its inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EgoState:
    x: float
    y: float
    heading: float          # radians, 0 = +x axis
    speed: float            # m/s
    width: float = 1.85
    length: float = 4.5


@dataclass
class Agent:
    x: float
    y: float
    vx: float
    vy: float
    width: float = 1.85
    length: float = 4.5


@dataclass
class Trajectory:
    waypoints: list[tuple[float, float]]    # (x, y) per timestep
    speeds: list[float]                     # m/s per timestep
    score: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_candidates(
    ego: EgoState,
    horizon: float,
    dt: float,
    lateral_offsets: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    target_speeds: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0),
) -> list[Trajectory]:
    """Constant-curvature, constant-acceleration roll-outs.

    For each (lateral_offset, target_speed) pair, generate a smooth lane
    change over the horizon and a linear acceleration to target speed.
    """
    n_steps = max(1, int(round(horizon / dt)))
    candidates: list[Trajectory] = []

    cos_h, sin_h = math.cos(ego.heading), math.sin(ego.heading)
    # Right-hand normal to heading (for lateral offset)
    nx, ny = -sin_h, cos_h

    for offset in lateral_offsets:
        for v_target in target_speeds:
            wpts: list[tuple[float, float]] = []
            speeds: list[float] = []
            v = ego.speed
            # Linear acceleration to target speed over the horizon
            a = (v_target - ego.speed) / horizon if horizon > 0 else 0.0
            s = 0.0  # arclength along heading
            for i in range(1, n_steps + 1):
                t = i * dt
                v = ego.speed + a * t
                # Smooth lateral interpolation: offset * smoothstep(t/horizon)
                u = min(1.0, t / horizon) if horizon > 0 else 1.0
                lat = offset * (3 * u * u - 2 * u * u * u)  # smoothstep
                s += max(0.0, v) * dt
                x = ego.x + cos_h * s + nx * lat
                y = ego.y + sin_h * s + ny * lat
                wpts.append((x, y))
                speeds.append(v)
            candidates.append(Trajectory(waypoints=wpts, speeds=speeds))

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _box_overlap(
    cx1: float, cy1: float, w1: float, l1: float,
    cx2: float, cy2: float, w2: float, l2: float,
) -> bool:
    """Axis-aligned bounding-box overlap test.

    A coarse approximation — adequate for cascade-signal measurement on mini
    where the magnitudes of degradation matter more than collision physics.
    """
    return (
        abs(cx1 - cx2) * 2 < (l1 + l2) and
        abs(cy1 - cy2) * 2 < (w1 + w2)
    )


def _score(
    traj: Trajectory,
    ego: EgoState,
    predicted_agents: list[Agent],
    dt: float,
    weights: dict[str, float],
) -> Trajectory:
    """Populate traj.score and traj.breakdown."""
    collision_pen = 0.0
    progress = 0.0
    comfort_pen = 0.0

    if traj.waypoints:
        # Progress: distance from start to last waypoint along ego heading
        dx = traj.waypoints[-1][0] - ego.x
        dy = traj.waypoints[-1][1] - ego.y
        progress = dx * math.cos(ego.heading) + dy * math.sin(ego.heading)

    # Comfort: penalise large speed changes between consecutive steps
    for i in range(1, len(traj.speeds)):
        comfort_pen += (traj.speeds[i] - traj.speeds[i - 1]) ** 2

    # Collision: at each timestep, check ego box vs each agent's predicted box
    for step_idx, (ex, ey) in enumerate(traj.waypoints):
        t = (step_idx + 1) * dt
        for ag in predicted_agents:
            ax = ag.x + ag.vx * t
            ay = ag.y + ag.vy * t
            if _box_overlap(ex, ey, ego.width, ego.length,
                            ax, ay, ag.width, ag.length):
                collision_pen += 1.0
                break  # one collision per timestep is enough

    score = (
        - weights.get("collision", 1000.0) * collision_pen
        + weights.get("progress", 1.0) * progress
        - weights.get("comfort", 0.1) * comfort_pen
    )

    traj.score = score
    traj.breakdown = {
        "collision_pen": collision_pen,
        "progress": progress,
        "comfort_pen": comfort_pen,
    }
    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def idm_plan(
    ego: EgoState,
    predicted_agents: list[Agent],
    time_horizon: float = 3.0,
    dt: float = 0.5,
    weights: dict[str, float] | None = None,
) -> Trajectory:
    """Score a candidate-trajectory grid, return the best.

    Returns:
        The argmax Trajectory. waypoints are (x, y) over horizon at dt steps.
    """
    weights = weights or {"collision": 1000.0, "progress": 1.0, "comfort": 0.1}
    cands = _generate_candidates(ego, time_horizon, dt)
    cands = [_score(t, ego, predicted_agents, dt, weights) for t in cands]
    return max(cands, key=lambda t: t.score)


def evaluate_c3(
    c2_checkpoint: str | None,
    split: str,
    mode: str,
    c1_checkpoint: str | None = None,
    splits_json: str | None = None,
    nusc=None,
) -> dict:
    """Run IDM planner over a split, return planning metrics.

    mode='isolated': planner sees GT future positions of all agents.
    mode='pipeline': planner sees C2's predictions, which come from C1
                     detections under the active perturbation.
    """
    import json as _json
    if nusc is None or splits_json is None:
        return {
            "L2@1s": None, "L2@2s": None, "L2@3s": None,
            "collision_rate": None, "n_scenes": 0,
            "mode": mode, "note": "skipped (no nusc)",
        }

    from .train_c2 import c2_predict, _predict_from_gt

    with open(splits_json) as f:
        splits = _json.load(f)

    l2_per_horizon = {1.0: [], 2.0: [], 3.0: []}
    collisions = 0
    n_scenes = 0

    for scene_tok in splits.get(split, []):
        n_scenes += 1
        if mode == "isolated":
            agent_preds = _predict_from_gt(scene_tok, nusc)
        elif mode == "pipeline":
            agent_preds = c2_predict(
                scene_tok, nusc, mode="pipeline",
                c1_descriptor_path=c1_checkpoint,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        agents = [
            Agent(x=ap.initial_pos[0], y=ap.initial_pos[1],
                  vx=ap.initial_vel[0], vy=ap.initial_vel[1])
            for ap in agent_preds
        ]

        scene = nusc.get("scene", scene_tok)
        sample = nusc.get("sample", scene["first_sample_token"])
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        ego_speed = 0.0
        if sample["next"]:
            sample2 = nusc.get("sample", sample["next"])
            sd2 = nusc.get("sample_data", sample2["data"]["LIDAR_TOP"])
            ep2 = nusc.get("ego_pose", sd2["ego_pose_token"])
            dt = (sample2["timestamp"] - sample["timestamp"]) / 1e6
            if dt > 0:
                dx = ep2["translation"][0] - ep["translation"][0]
                dy = ep2["translation"][1] - ep["translation"][1]
                ego_speed = math.hypot(dx, dy) / dt
        ego = EgoState(x=0.0, y=0.0, heading=0.0, speed=ego_speed)

        traj = idm_plan(ego, agents, time_horizon=3.0, dt=0.5)

        gt_pts = _ego_future(nusc, scene_tok, n_steps=6, dt=0.5)
        for h_sec, idx in ((1.0, 1), (2.0, 3), (3.0, 5)):
            if idx >= len(traj.waypoints) or idx >= len(gt_pts):
                continue
            px, py = traj.waypoints[idx]
            gx, gy = gt_pts[idx]
            l2_per_horizon[h_sec].append(math.hypot(px - gx, py - gy))

        if traj.breakdown.get("collision_pen", 0) > 0:
            collisions += 1

    def _avg(xs): return (sum(xs) / len(xs)) if xs else None
    return {
        "L2@1s": _avg(l2_per_horizon[1.0]),
        "L2@2s": _avg(l2_per_horizon[2.0]),
        "L2@3s": _avg(l2_per_horizon[3.0]),
        "collision_rate": (collisions / n_scenes) if n_scenes else 0.0,
        "n_scenes": n_scenes,
        "mode": mode,
    }


def _ego_future(nusc, scene_token: str, n_steps: int = 6, dt: float = 0.5):
    """Ego positions in heading-aligned ego-frame of first sample.

    The planner operates with x = forward along the ego heading at t=0.
    To compare planned trajectory to the human-driver trajectory, the
    nuScenes global poses must be rotated by -heading_t0.
    """
    scene = nusc.get("scene", scene_token)
    sample0 = nusc.get("sample", scene["first_sample_token"])
    sd0 = nusc.get("sample_data", sample0["data"]["LIDAR_TOP"])
    ep0 = nusc.get("ego_pose", sd0["ego_pose_token"])
    t0 = sample0["timestamp"] / 1e6
    x0, y0, _ = ep0["translation"]

    # Compute initial heading from quaternion (w, x, y, z): yaw = atan2(2(wz+xy), 1-2(yy+zz))
    qw, qx, qy, qz = ep0["rotation"]
    heading0 = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cos_h, sin_h = math.cos(-heading0), math.sin(-heading0)

    points = []
    sample_tok = sample0["token"]
    while sample_tok:
        s = nusc.get("sample", sample_tok)
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        t = s["timestamp"] / 1e6
        ex, ey, _ = ep["translation"]
        # Translate then rotate into ego-frame at t=0
        dx, dy = ex - x0, ey - y0
        ego_x = cos_h * dx - sin_h * dy
        ego_y = sin_h * dx + cos_h * dy
        points.append((t - t0, ego_x, ego_y))
        sample_tok = s["next"]

    out = []
    for i in range(1, n_steps + 1):
        target_t = i * dt
        for j in range(len(points) - 1):
            if points[j][0] <= target_t <= points[j + 1][0]:
                t_a, xa, ya = points[j]
                t_b, xb, yb = points[j + 1]
                u = (target_t - t_a) / (t_b - t_a) if t_b > t_a else 0
                out.append((xa + u * (xb - xa), ya + u * (yb - ya)))
                break
        else:
            if points:
                out.append((points[-1][1], points[-1][2]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Offline self-test — no Kaggle, no dataset.
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    # 1. Empty scene → planner picks the candidate with most progress.
    ego = EgoState(x=0.0, y=0.0, heading=0.0, speed=10.0)
    traj = idm_plan(ego, predicted_agents=[])
    assert traj.breakdown["collision_pen"] == 0.0
    assert traj.breakdown["progress"] > 0.0
    print(f"empty-scene best progress: {traj.breakdown['progress']:.2f} m")

    # 2. Agent dead ahead → planner should pick a non-zero lateral offset
    #    or slow to a stop. Place the blocker far enough that a smooth
    #    lane change clears it before any timestep overlap.
    blocker = Agent(x=25.0, y=0.0, vx=0.0, vy=0.0)
    traj = idm_plan(ego, predicted_agents=[blocker])
    assert traj.breakdown["collision_pen"] == 0.0, (
        f"failed to avoid blocker: {traj.breakdown}"
    )
    final_y = traj.waypoints[-1][1]
    assert abs(final_y) > 0.5, f"expected lateral offset, got y={final_y}"
    print(f"avoidance ok — final waypoint: {traj.waypoints[-1]}")

    # 3. Determinism: same inputs → same output.
    t1 = idm_plan(ego, [blocker])
    t2 = idm_plan(ego, [blocker])
    assert t1.waypoints == t2.waypoints
    print("deterministic OK")

    print("self-test OK")


if __name__ == "__main__":
    _self_test()
