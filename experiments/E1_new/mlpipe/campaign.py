"""Campaign evaluation: one C1 update through the saved C2/C3 (no retraining).

For an update (base old C1 -> updated C1) it reports, on singapore_test:
  H1: delta2 (prediction error) and delta3 (planning error, both C3 variants) for
      C2/C3 trained on the ground truth (isolated) and on the base old C1 (old-pipeline)
  H2: the label-free change of the C1 output (Table 4) and of the C2 output
The random-variation reference is the same comparison with the other old C1 seed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from . import data as D
from .evaluate import c2_metrics, c2_output_change, c3_metrics, object_use
from .models import C2AutoBot, C3PlanT
from .train import c3_objects_from_c2, c3_objects_from_gt, load_model, plan_c3, predict_c2

MATCH_RADIUS_M = 2.0
C3_VARIANTS = {"ego": True, "noego": False}


# ───────────────────────── H2 measures on the C1 output ─────────────────────────

def _js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)


def _only_one_version_fraction(a_by, b_by, images):
    unmatched = total = 0
    for img in images:
        a, b = a_by.get(img, []), b_by.get(img, [])
        total += len(a) + len(b); used = set()
        for da in a:
            best, bd = None, MATCH_RADIUS_M
            for j, db in enumerate(b):
                if j in used or db[0] != da[0]:
                    continue
                d = float(np.hypot(da[1] - db[1], da[2] - db[2]))
                if d <= bd: best, bd = j, d
            if best is None: unmatched += 1
            else: used.add(best)
        unmatched += len(b) - len(used)
    return unmatched / total if total else 0.0


def h2_c1(det_a: dict, det_b: dict) -> dict:
    """det_*: sample_token -> [[cls, x, y, score], ...] (ego frame). No ground truth used."""
    images = sorted(set(det_a) | set(det_b))
    ca = [len(det_a.get(i, [])) for i in images]; cb = [len(det_b.get(i, [])) for i in images]
    m = max(ca + cb) + 1
    ra = [r for i in images for r in det_a.get(i, [])]; rb = [r for i in images for r in det_b.get(i, [])]
    ka, kb = Counter(int(r[0]) for r in ra), Counter(int(r[0]) for r in rb)
    return {
        "objects_per_image_JS": _js(np.bincount(ca, minlength=m) + 1e-9, np.bincount(cb, minlength=m) + 1e-9),
        "mean_objects_per_image": [float(np.mean(ca)), float(np.mean(cb))],
        "type_mix_JS": _js([ka[c] + 1e-9 for c in range(D.N_CLS)], [kb[c] + 1e-9 for c in range(D.N_CLS)]),
        "confidence_W": float(wasserstein_distance([r[3] for r in ra], [r[3] for r in rb])),
        "confidence_KS": float(ks_2samp([r[3] for r in ra], [r[3] for r in rb]).statistic),
        "forward_position_W_m": float(wasserstein_distance([r[1] for r in ra], [r[1] for r in rb])),
        "lateral_position_W_m": float(wasserstein_distance([r[2] for r in ra], [r[2] for r in rb])),
        "only_one_version_fraction": _only_one_version_fraction(det_a, det_b, images),
    }


# ───────────────────────── H1 for one update ─────────────────────────

class Evaluator:
    """Holds the saved C2/C3 for one base old C1 seed and caches the base predictions."""

    def __init__(self, export: dict, models_dir: Path, base_seed: int, log=print):
        self.ex, self.log = export, log
        X = base_seed
        self.base, self.other = f"c1_old_s{X}", f"c1_old_s{1 - X}"
        mdir = Path(models_dir)
        stems = {"isolated": "p1_isolated_seed0", "old_pipeline": f"p1_old_pipeline_c1s{X}_seed0"}
        self.models = {}
        for name, stem in stems.items():
            c2 = load_model(C2AutoBot, mdir / f"{stem}_c2.pt")
            c3 = {v: load_model(C3PlanT, mdir / f"{stem}_c3_{v}.pt", use_ego=e) for v, e in C3_VARIANTS.items()}
            self.models[name] = (c2, c3)
        self.cache = {}
        self.gt = D.build_samples(export, "gt")
        self.anchors = [s.anchor for s in self.gt]
        self.iso = {}
        for name, (c2, c3) in self.models.items():
            objs = [c3_objects_from_gt(s) for s in self.gt]
            self.iso[name] = {"c2": c2_metrics(self.gt, predict_c2(c2, self.gt)),
                              "c3": {v: c3_metrics(self.gt, plan_c3(m, objs, self.gt)) for v, m in c3.items()}}

    def _pipe(self, setting: str, c1: str):
        key = (setting, c1)
        if key not in self.cache:
            c2, c3 = self.models[setting]
            S = D.build_samples(self.ex, f"det:{c1}")
            assert [s.anchor for s in S] == self.anchors
            P = predict_c2(c2, S)
            objs = [c3_objects_from_c2(s, p) for s, p in zip(S, P)]
            r = {"c2": c2_metrics(S, P), "c3": {}}
            for v, m in c3.items():
                plans = plan_c3(m, objs, S)
                r["c3"][v] = {**c3_metrics(S, plans),
                              "object_use_m": object_use(plans, plan_c3(m, objs, S, drop_objects=True))}
            self.cache[key] = (S, P, r)
        return self.cache[key]

    def update(self, upd: str) -> dict:
        out = {"base": self.base, "update": upd,
               "h2_c1_update": h2_c1(self.ex["detections"][self.base], self.ex["detections"][upd]),
               "settings": {}}
        for setting in self.models:
            Sb, Pb, rb = self._pipe(setting, self.base)
            Su, Pu, ru = self._pipe(setting, upd)
            res = {"base": rb, "updated": ru,
                   "delta2_minADE": ru["c2"]["minADE"] - rb["c2"]["minADE"],
                   "delta3_L2@3s": {v: ru["c3"][v]["L2@3s"] - rb["c3"][v]["L2@3s"] for v in C3_VARIANTS},
                   "delta_collision": {v: ru["c3"][v]["collision_rate"] - rb["c3"][v]["collision_rate"]
                                       for v in C3_VARIANTS},
                   "h2_c2_update": c2_output_change(Sb, Pb, Su, Pu)}
            out["settings"][setting] = res
        return out

    def seed_noise(self) -> dict:
        """Random-variation reference: base old C1 vs the other old C1 seed."""
        r = self.update(self.other)
        r["note"] = "random variation: only the old C1 training seed differs"
        return r
