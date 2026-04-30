"""C3 — Rule-based IDM planner (PDM-Closed substitute for free-tier compute).

Captures the cascade mechanism without nuPlan dependencies.
"""

from dataclasses import dataclass


@dataclass
class EgoState:
    x: float
    y: float
    heading: float
    speed: float


@dataclass
class Agent:
    x: float
    y: float
    vx: float
    vy: float
    width: float
    length: float


def idm_plan(
    ego: EgoState,
    predicted_agents: list[Agent],
    time_horizon: float = 3.0,
    dt: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Score candidate trajectories (lateral offsets + speed profiles).

    Returns list of (x, y) waypoints over time_horizon seconds.
    """
    # TODO: enumerate candidates over (lateral_offset, target_speed) grid
    # TODO: for each, simulate forward, compute collision_penalty + comfort + progress
    # TODO: return arg-max trajectory
    raise NotImplementedError("placeholder")


def evaluate_c3(
    c2_checkpoint: str | None,
    split: str,
    mode: str,
    c1_checkpoint: str | None = None,
) -> dict:
    """
    Run IDM planner over scenes in split, return {'L2@1s', 'L2@2s', 'L2@3s', 'collision_rate'}.

    mode='isolated': feed ground-truth predictions to the planner
    mode='pipeline': feed C2's predictions (from c1's detections if c1_checkpoint given)
    """
    # TODO: load split scenes; for each scene, get predictions per mode
    # TODO: call idm_plan; accumulate L2 vs human trajectory + collisions
    raise NotImplementedError("placeholder")
