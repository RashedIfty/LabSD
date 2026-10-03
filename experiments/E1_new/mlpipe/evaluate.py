"""Metrics for C2 and C3 (all on CPU, deterministic).

prediction error : minADE over the 6 modes, on objects of keyframe i that have
                   a GT future (for detections: assigned to a GT object)
planning error   : L2 between the planned path and the human path at 1, 2, 3 s
                   (the point convention of the earlier experiments), mean over anchors
collision rate   : fraction of anchors where the planned ego box overlaps a GT
                   object box at any of the 6 future steps (oriented boxes)
object use       : mean change of the 3 s waypoint when the objects are removed
                   from the input of C3
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import wasserstein_distance

EGO_L, EGO_W = 4.08, 1.73       # nuScenes data-collection vehicle (Renault Zoe)


def c2_metrics(samples, preds):
    ades, n_false, n_matched, n_vis = [], 0, 0, 0
    for s, (mu, _) in zip(samples, preds):
        n_vis += s.n_gt_current_visible
        for q in range(len(s.current)):
            m = s.fut_mask[q]
            if not m.any():
                n_false += 1; continue
            n_matched += 1
            err = np.linalg.norm(mu[q] - s.fut[q][None], axis=-1)         # [M,T]
            ades.append(float((err[:, m].mean(-1)).min()))
    return {"minADE": float(np.mean(ades)) if ades else float("nan"),
            "n_objects_scored": n_matched, "n_objects_without_gt": n_false,
            "n_gt_visible": n_vis}


def _corners(x, y, yaw, l, w):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2)]
    return np.array([(x + c * a - s * b, y + s * a + c * b) for a, b in pts])


def _overlap(p, q):
    """Separating axis test for two convex quadrilaterals."""
    for poly in (p, q):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def c3_metrics(samples, plans):
    l2 = {1: [], 2: [], 3: []}
    coll = 0
    for s, wp in zip(samples, plans):
        for sec, k in ((1, 1), (2, 3), (3, 5)):
            l2[sec].append(float(np.linalg.norm(wp[k] - s.ego_fut[k])))
        prev = np.zeros(2); hit = False
        for t in range(len(wp)):
            d = wp[t] - prev
            yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 0.1 else 0.0
            prev = wp[t]
            ego = _corners(wp[t][0], wp[t][1], yaw, EGO_L, EGO_W)
            for ox, oy, oyaw, ow, ol in s.gt_objects_future[t]:
                if abs(ox - wp[t][0]) > 12 or abs(oy - wp[t][1]) > 12:
                    continue
                if _overlap(ego, _corners(ox, oy, oyaw, ol, ow)):
                    hit = True; break
            if hit: break
        coll += hit
    n = len(samples)
    return {"L2@1s": float(np.mean(l2[1])), "L2@2s": float(np.mean(l2[2])),
            "L2@3s": float(np.mean(l2[3])), "collision_rate": coll / n if n else float("nan"),
            "n_anchors": n}


def object_use(plans_with, plans_without):
    return float(np.linalg.norm(plans_with[:, -1] - plans_without[:, -1], axis=-1).mean())


def c2_output_change(samples_a, preds_a, samples_b, preds_b):
    """H2 measures on the C2 output between two pipelines on the same anchors.

    movement_W   : Wasserstein distance between the predicted 3 s movement lengths
                   (most likely mode) of all objects
    same_object_shift : mean distance between the 3 s predicted positions of
                   objects that both versions found (same class, nearest within 2 m)
    """
    mv = lambda S, P: [float(np.linalg.norm(mu[q, pr[q].argmax(), -1] - s.cur_xy[q]))
                       for s, (mu, pr) in zip(S, P) for q in range(len(s.current))]
    a, b = mv(samples_a, preds_a), mv(samples_b, preds_b)
    shifts = []
    for sa, (ma, pa), sb, (mb, pb) in zip(samples_a, preds_a, samples_b, preds_b):
        assert sa.anchor == sb.anchor
        used = set()
        for qa in range(len(sa.current)):
            best, bd = None, 2.0
            for qb in range(len(sb.current)):
                if qb in used or sb.cur_cls[qb] != sa.cur_cls[qa]:
                    continue
                d = float(np.linalg.norm(sa.cur_xy[qa] - sb.cur_xy[qb]))
                if d <= bd: best, bd = qb, d
            if best is None: continue
            used.add(best)
            shifts.append(float(np.linalg.norm(ma[qa, pa[qa].argmax(), -1] - mb[best, pb[best].argmax(), -1])))
    return {"movement_W": float(wasserstein_distance(a, b)) if a and b else float("nan"),
            "same_object_shift": float(np.mean(shifts)) if shifts else float("nan"),
            "n_same_objects": len(shifts)}
