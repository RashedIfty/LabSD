"""Paired scene bootstrap of delta2 and delta3 for each C1 update.

The 50 test scenes are resampled with replacement; for every resample the
prediction error (mean over scored objects) and the planning error (mean over
anchors) are recomputed for the base old C1 and the updated C1 on the same
scenes, and their difference is taken. The 2.5 and 97.5 percentiles give a 95 %
interval. An update counts as harmful when the whole interval is above zero.
"""
from __future__ import annotations

import numpy as np

from . import data as D
from .train import c3_objects_from_c2, plan_c3, predict_c2


def scene_sums(ev, setting: str, c1: str, variant: str = "ego"):
    """Per scene: (sum of object minADE, number of objects, sum of L2@3s, number of anchors)."""
    c2, c3 = ev.models[setting]
    S = D.build_samples(ev.ex, f"det:{c1}")
    P = predict_c2(c2, S)
    plans = plan_c3(c3[variant], [c3_objects_from_c2(s, p) for s, p in zip(S, P)], S)
    acc = {}
    for s, (mu, _), wp in zip(S, P, plans):
        a = acc.setdefault(s.scene, [0.0, 0, 0.0, 0])
        for q in range(len(s.current)):
            m = s.fut_mask[q]
            if m.any():
                a[0] += float(np.linalg.norm(mu[q] - s.fut[q][None], axis=-1)[:, m].mean(-1).min())
                a[1] += 1
        a[2] += float(np.linalg.norm(wp[5] - s.ego_fut[5])); a[3] += 1
    return acc


def paired_bootstrap(base: dict, upd: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    scenes = sorted(set(base) | set(upd))
    B = np.array([base.get(s, [0, 0, 0, 0]) for s in scenes], float)
    U = np.array([upd.get(s, [0, 0, 0, 0]) for s in scenes], float)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(scenes), np.ones(len(scenes)) / len(scenes), size=n_boot).astype(float)
    def delta(cs, cn):
        return (W @ U[:, cs]) / np.maximum(W @ U[:, cn], 1) - (W @ B[:, cs]) / np.maximum(W @ B[:, cn], 1)
    out = {}
    for name, cs, cn in (("delta2_minADE", 0, 1), ("delta3_L2@3s", 2, 3)):
        point = U[:, cs].sum() / U[:, cn].sum() - B[:, cs].sum() / B[:, cn].sum()
        d = delta(cs, cn)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = {"point": float(point), "ci95": [float(lo), float(hi)],
                     "harm": bool(lo > 0), "benefit": bool(hi < 0)}
    out["n_scenes"] = len(scenes)
    return out
