"""Interface drift gate between C1 (perception) and C2 (prediction).

Phase-2a of SENSEI_FEEDBACK_PLAN.md (sensei Q2 — evaluate the proposed method).
This is the concrete front-end of CARA: before a retrained detector is admitted
into the pipeline, compare the statistical profile of its outputs against the
incumbent detector's outputs on a fixed audit set. If the profiles differ too
much, raise a "cascade risk" flag.

The gate is deliberately cheap and model-free: it works only on the detection
records the pipeline already produces (class, ego-frame position, confidence),
using two standard two-sample tests:
  * Kolmogorov-Smirnov (KS) statistic for continuous features
    (confidence, box-center x, box-center y, per-image detection count);
  * Jensen-Shannon (JS) divergence for the categorical feature (class mix).

A single scalar drift score aggregates these; a threshold tau turns it into a
binary admit / cascade-risk decision. Nothing here imports torch — it consumes
lists of detection dicts, so it is fully testable offline and can be run on the
detection dumps collected from any campaign.
"""
from __future__ import annotations

import math
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from detection records
# ─────────────────────────────────────────────────────────────────────────────

# A "detection dump" is a list of records, each:
#   {"sample_token": str, "cls": str, "x": float, "y": float, "score": float}
# (exactly the fields c1_detect_yolo emits).

_CONTINUOUS = ("score", "x", "y")
_CLASSES = ("car", "truck", "bus", "trailer", "construction_vehicle",
            "motorcycle", "bicycle", "pedestrian")


def _per_image_counts(dets: list[dict]) -> list[float]:
    """Number of detections per image (grouped by sample_token)."""
    c = Counter(d["sample_token"] for d in dets)
    return [float(v) for v in c.values()] or [0.0]


def _class_hist(dets: list[dict]) -> list[float]:
    """Normalised class histogram over the fixed class list."""
    c = Counter(d.get("cls") for d in dets)
    total = sum(c.get(k, 0) for k in _CLASSES)
    if total == 0:
        return [1.0 / len(_CLASSES)] * len(_CLASSES)  # uniform if empty
    return [c.get(k, 0) / total for k in _CLASSES]


def _values(dets: list[dict], key: str) -> list[float]:
    return [float(d[key]) for d in dets if key in d] or [0.0]


# ─────────────────────────────────────────────────────────────────────────────
# Two-sample tests (self-contained; no scipy required)
# ─────────────────────────────────────────────────────────────────────────────

def ks_statistic(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic in [0,1].

    Max vertical gap between the two empirical CDFs. 0 = identical
    distributions, 1 = fully separated.
    """
    if not a or not b:
        return 0.0
    xs = sorted(set(a) | set(b))
    na, nb = len(a), len(b)
    sa = sorted(a)
    sb = sorted(b)

    def cdf(sorted_vals, x, n):
        # fraction of samples <= x
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_vals[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / n

    return max(abs(cdf(sa, x, na) - cdf(sb, x, nb)) for x in xs)


def js_divergence(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence between two categorical distributions.

    Returned in [0,1] (base-2 logs). 0 = identical mixes, 1 = disjoint.
    """
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")

    def _kl(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                s += ai * math.log2(ai / bi)
        return s

    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ─────────────────────────────────────────────────────────────────────────────
# The gate
# ─────────────────────────────────────────────────────────────────────────────

def drift_profile(old_dets: list[dict], new_dets: list[dict]) -> dict:
    """Per-feature drift between two detector outputs on the same audit set.

    Returns each feature's two-sample statistic plus an aggregate `drift`
    score (mean of the per-feature statistics, all in [0,1]).
    """
    feats: dict[str, float] = {}
    # continuous features -> KS
    feats["count"] = ks_statistic(_per_image_counts(old_dets),
                                  _per_image_counts(new_dets))
    for key in _CONTINUOUS:
        feats[key] = ks_statistic(_values(old_dets, key), _values(new_dets, key))
    # categorical feature -> JS
    feats["class"] = js_divergence(_class_hist(old_dets), _class_hist(new_dets))

    drift = sum(feats.values()) / len(feats)
    return {"per_feature": feats, "drift": drift}


def gate_decision(old_dets: list[dict], new_dets: list[dict],
                  tau: float = 0.25) -> dict:
    """Admit / cascade-risk decision for a retrained detector.

    tau is the drift threshold. drift <= tau -> admit; else cascade risk.
    Returns the profile, the scalar drift, tau, and the boolean flag.
    """
    prof = drift_profile(old_dets, new_dets)
    flag = prof["drift"] > tau
    return {
        "drift": prof["drift"],
        "per_feature": prof["per_feature"],
        "tau": tau,
        "cascade_risk": flag,      # True = hold / inspect; False = admit
    }
