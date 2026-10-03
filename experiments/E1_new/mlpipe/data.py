"""Samples for the ML prediction module C2 and the ML planning module C3.

Input: the Stage 1 export (one JSON per split) with, per keyframe, the global
ego pose of the CAM_FRONT sample, the ground-truth (GT) objects in global
coordinates, and the detections of each C1 model in the ego frame of that
keyframe (x forward, y left).

One sample is built at every keyframe i (the anchor) that has 6 later
keyframes (3 s at 2 Hz). Everything is expressed in the ego frame at i.

C2 input:  the objects of keyframes i-3..i (2 s) as an unordered set of tokens
           (no track identities). Source: GT objects visible in CAM_FRONT
           ("gt"), or the detections of one C1 model ("det:<model>").
C2 target: for every object of keyframe i, the GT future at i+1..i+6 with a
           validity mask. A detection gets the future of the GT object it is
           assigned to (one-to-one Hungarian assignment, same class, within a
           range-dependent radius). The assignment only builds labels and
           metrics; it is not part of the pipeline.
C3 target: the ego positions at i+1..i+6 (the path the human driver drove).
Command:   left / straight / right from the lateral ego offset at 3 s
           (+-2 m), the rule used by VAD and AD-MLP.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

N_CLS = 8
HIST = 4            # keyframes i-3..i
FUT = 6             # keyframes i+1..i+6
DT = 0.5
DT_TOL = 0.15       # a keyframe step must be 0.5 s +- 0.15 s
TOKEN_DIM = 2 + N_CLS + 1 + 1   # x, y, class one-hot, confidence, time offset


def assign_radius(rng: float) -> float:
    """Assignment radius (m): 2 m near the ego vehicle, growing with range.

    Back-projected camera detections are less precise in depth far away.
    """
    return max(2.0, 0.15 * rng)


def to_local(xy: np.ndarray, ego: tuple[float, float, float]) -> np.ndarray:
    """Global (x, y) -> ego frame of pose (ex, ey, yaw)."""
    ex, ey, yaw = ego
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(xy, float) - np.array([ex, ey])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def to_global(xy: np.ndarray, ego: tuple[float, float, float]) -> np.ndarray:
    ex, ey, yaw = ego
    c, s = math.cos(yaw), math.sin(yaw)
    xy = np.asarray(xy, float)
    return np.stack([c * xy[..., 0] - s * xy[..., 1] + ex, s * xy[..., 0] + c * xy[..., 1] + ey], -1)


def command_of(ego_future: np.ndarray) -> int:
    """0 = left, 1 = straight, 2 = right (lateral offset at 3 s)."""
    y = ego_future[-1, 1]
    return 0 if y >= 2.0 else (2 if y <= -2.0 else 1)


@dataclass
class Sample:
    scene: str
    anchor: str                       # sample token of keyframe i
    tokens: np.ndarray                # [N, TOKEN_DIM] C2 input set
    current: np.ndarray               # [Q] indices into tokens: objects of keyframe i
    cur_xy: np.ndarray                # [Q, 2]
    cur_cls: np.ndarray               # [Q]
    cur_conf: np.ndarray              # [Q]
    fut: np.ndarray                   # [Q, FUT, 2] GT future of each current object (ego frame i)
    fut_mask: np.ndarray              # [Q, FUT] bool
    ego_fut: np.ndarray               # [FUT, 2] human ego path
    command: int
    gt_objects_future: list = field(default_factory=list)  # per step: [M, 5] x, y, yaw, w, l (ego frame i), for collisions
    n_gt_current_visible: int = 0     # GT objects visible in CAM_FRONT at keyframe i (for miss counts)
    ego_hist: np.ndarray = None       # [HIST-1, 2] ego positions at i-1..i-3 (ego frame i), zeros if missing
    ego_hist_mask: np.ndarray = None  # [HIST-1] bool


def _objects_at(sample: dict, source: str, ego_ref, detections: dict | None):
    """Objects of one keyframe in the ego frame ego_ref: (xy [K,2], cls [K], conf [K], inst list or None)."""
    if source == "gt":
        rows = [g for g in sample["gt"] if g[7] == 1]
        if not rows:
            return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0), []
        xy = to_local(np.array([[g[2], g[3]] for g in rows]), ego_ref)
        return xy, np.array([g[1] for g in rows]), np.ones(len(rows)), [g[0] for g in rows]
    det = detections.get(sample["token"], [])
    if not det:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0), None
    d = np.array(det, float)
    xy_own = d[:, 1:3]
    xy = to_local(to_global(xy_own, sample["ego"]), ego_ref)
    return xy, d[:, 0].astype(int), d[:, 3], None


def _gt_future_by_instance(samples, i, ego_ref):
    """instance -> [FUT, 2] positions (ego frame i) and mask, plus per-step boxes for collisions."""
    fut = {}
    boxes = []
    for k in range(1, FUT + 1):
        s = samples[i + k]
        rows = s["gt"]
        if rows:
            xy = to_local(np.array([[g[2], g[3]] for g in rows]), ego_ref)
            yaw = np.array([g[4] for g in rows]) - ego_ref[2]
            boxes.append(np.column_stack([xy, yaw, [g[5] for g in rows], [g[6] for g in rows]]))
            for g, p in zip(rows, xy):
                arr = fut.setdefault(g[0], [np.zeros((FUT, 2)), np.zeros(FUT, bool)])
                arr[0][k - 1] = p
                arr[1][k - 1] = True
        else:
            boxes.append(np.zeros((0, 5)))
    return fut, boxes


def _assign(det_xy, det_cls, gt_xy, gt_cls):
    """One-to-one assignment detection -> GT index (or -1)."""
    out = -np.ones(len(det_xy), int)
    if len(det_xy) == 0 or len(gt_xy) == 0:
        return out
    d = np.linalg.norm(det_xy[:, None, :] - gt_xy[None, :, :], axis=-1)
    rad = np.array([assign_radius(float(np.hypot(*p))) for p in det_xy])[:, None]
    bad = (d > rad) | (det_cls[:, None] != gt_cls[None, :])
    cost = np.where(bad, 1e6, d)
    r, c = linear_sum_assignment(cost)
    for a, b in zip(r, c):
        if cost[a, b] < 1e6:
            out[a] = b
    return out


def build_samples(export: dict, source: str, scene_filter: set | None = None) -> list[Sample]:
    """All anchor samples of one split for one input source ("gt" or "det:<model>")."""
    detections = None
    if source.startswith("det:"):
        detections = export["detections"][source[4:]]
    out: list[Sample] = []
    for sc in export["scenes"]:
        if scene_filter is not None and sc["scene"] not in scene_filter:
            continue
        S = sc["samples"]
        for i in range(len(S) - FUT):
            # Time checks: 6 future steps of 0.5 s each
            t0 = S[i]["t"]
            if any(abs((S[i + k]["t"] - t0) / 1e6 - k * DT) > DT_TOL * k for k in range(1, FUT + 1)):
                continue
            ego_ref = tuple(S[i]["ego"])
            ego_fut = to_local(np.array([S[i + k]["ego"][:2] for k in range(1, FUT + 1)]), ego_ref)
            ego_hist = np.zeros((HIST - 1, 2)); ego_hist_mask = np.zeros(HIST - 1, bool)
            for h in range(1, HIST):
                if i - h >= 0 and abs((t0 - S[i - h]["t"]) / 1e6 - h * DT) <= DT_TOL * h:
                    ego_hist[h - 1] = to_local(np.array(S[i - h]["ego"][:2]), ego_ref)
                    ego_hist_mask[h - 1] = True

            # C2 input set: keyframes i-3..i
            toks, cur_idx = [], []
            cur = None
            for h in range(HIST - 1, -1, -1):
                j = i - h
                if j < 0:
                    continue
                dt = (S[j]["t"] - t0) / 1e6
                if dt < -(HIST - 1) * DT - DT_TOL * HIST:
                    continue
                xy, cls, conf, inst = _objects_at(S[j], source, ego_ref, detections)
                for k in range(len(xy)):
                    if h == 0:
                        cur_idx.append(len(toks))
                    oh = np.zeros(N_CLS); oh[cls[k]] = 1
                    toks.append(np.concatenate([xy[k], oh, [conf[k], dt]]))
                if h == 0:
                    cur = (xy, cls, conf, inst)
            if cur is None:
                continue
            cxy, ccls, cconf, cinst = cur
            gt_fut, boxes = _gt_future_by_instance(S, i, ego_ref)

            fut = np.zeros((len(cxy), FUT, 2))
            mask = np.zeros((len(cxy), FUT), bool)
            if source == "gt":
                for q, inst in enumerate(cinst):
                    if inst in gt_fut:
                        fut[q], mask[q] = gt_fut[inst]
            else:
                rows = S[i]["gt"]
                gxy = to_local(np.array([[g[2], g[3]] for g in rows]), ego_ref) if rows else np.zeros((0, 2))
                gcls = np.array([g[1] for g in rows], int)
                match = _assign(cxy, ccls, gxy, gcls)
                for q, m in enumerate(match):
                    if m >= 0 and rows[m][0] in gt_fut:
                        fut[q], mask[q] = gt_fut[rows[m][0]]

            out.append(Sample(
                scene=sc["scene"], anchor=S[i]["token"],
                tokens=np.array(toks, np.float32).reshape(-1, TOKEN_DIM),
                current=np.array(cur_idx, int), cur_xy=cxy.astype(np.float32),
                cur_cls=ccls.astype(int), cur_conf=cconf.astype(np.float32),
                fut=fut.astype(np.float32), fut_mask=mask,
                ego_fut=ego_fut.astype(np.float32), command=command_of(ego_fut),
                gt_objects_future=boxes,
                n_gt_current_visible=sum(1 for g in S[i]["gt"] if g[7] == 1),
                ego_hist=ego_hist.astype(np.float32), ego_hist_mask=ego_hist_mask,
            ))
    return out


def load_export(path: str | Path) -> dict:
    return json.load(open(path))


def split_train_val(export: dict, every: int = 7) -> tuple[set, set]:
    """Deterministic scene split of a training export: every 7th scene (name order) is validation."""
    names = sorted((sc["name"], sc["scene"]) for sc in export["scenes"])
    val = {tok for k, (_, tok) in enumerate(names) if k % every == every - 1}
    return {tok for _, tok in names} - val, val


def augment(s: "Sample", rng: np.random.Generator, max_rot_deg: float = 10.0) -> "Sample":
    """Random mirror (left-right) and small rotation about the ego vehicle, applied to every position."""
    import copy
    a = math.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    flip = rng.random() < 0.5
    c, sn = math.cos(a), math.sin(a)
    def tf(xy):
        xy = np.array(xy, np.float32, copy=True)
        if flip: xy[..., 1] = -xy[..., 1]
        x, y = xy[..., 0].copy(), xy[..., 1].copy()
        xy[..., 0], xy[..., 1] = c * x - sn * y, sn * x + c * y
        return xy
    o = copy.copy(s)
    if len(s.tokens):
        o.tokens = s.tokens.copy(); o.tokens[:, :2] = tf(s.tokens[:, :2])
    o.cur_xy = tf(s.cur_xy); o.fut = tf(s.fut); o.ego_fut = tf(s.ego_fut)
    o.ego_hist = tf(s.ego_hist)
    o.command = (2 - s.command) if flip else s.command
    o._aug = (flip, a)
    return o
