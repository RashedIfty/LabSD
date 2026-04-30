"""Five-measurement harness producing one row each for Table I.

The five measurements (Meeting 3 report, §VI Table I):
    1) C1 mAP on Singapore                    — c1_mAP
    2) C2 minADE — isolated (GT detections)   — c2_iso_minADE
    3) C2 minADE — pipeline (live C1 dets)    — c2_pipe_minADE
    4) C3 L2    — isolated (GT predictions)   — c3_iso_L2
    5) C3 L2    — pipeline (live C2 preds)    — c3_pipe_L2

Plus an auxiliary safety metric:
    6) C3 collision rate — pipeline           — c3_pipe_collision_rate

This module is a thin orchestrator. The heavy lifting lives in:
    train_c1.evaluate_c1
    train_c2.evaluate_c2
    c3_idm.evaluate_c3

Each of those is GPU-bound and currently stubbed. To let us validate the
table/diagnostic plumbing today, eval.py supports a `mock=True` mode that
returns plausible numbers — useful for testing locally before any Kaggle
run produces real metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Mock metric source — used until the real evaluators are wired up on Kaggle.
# Returned numbers are illustrative, not predictive: they roughly mimic what
# the Meeting 3 report expects to see (C3 pipeline worsens, C3 isolated stays).
# ─────────────────────────────────────────────────────────────────────────────

def _mock_metrics(retrain_c1: bool, split: str) -> dict:
    """Return a plausible-but-fake measurement dict.

    The trick: with retrain_c1=True we worsen pipeline metrics slightly to
    simulate cascade. Isolated metrics never change. C1 mAP improves (that's
    the point of retraining).
    """
    base = {
        "c1_mAP":                 0.42,
        "c2_iso_minADE":          1.20,
        "c2_pipe_minADE":         1.45,
        "c3_iso_L2":              1.10,
        "c3_pipe_L2":             1.55,
        "c3_pipe_collision_rate": 0.08,
    }
    if retrain_c1:
        # C1 improves on Singapore; pipeline metrics for C2 and C3 drift up;
        # isolated metrics held constant by construction.
        base = dict(base)
        base["c1_mAP"]                 = 0.51
        base["c2_pipe_minADE"]         = 1.62
        base["c3_pipe_L2"]             = 1.83
        base["c3_pipe_collision_rate"] = 0.11
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_all_measurements(
    c1_ckpt: str,
    c2_ckpt: str,
    split: str,
    out_path: str,
    mock: bool = False,
    retrain_c1_for_mock: bool = False,
    nusc=None,
    splits_json: str | None = None,
) -> dict:
    """Compute the five (+1) values and write them to JSON.

    Args:
        c1_ckpt, c2_ckpt: paths to component checkpoints.
        split: which val split to evaluate on (e.g. 'singapore_val').
        out_path: where to write the JSON.
        mock: if True, skip real evaluators and emit synthetic metrics.
        retrain_c1_for_mock: only used when mock=True.
        nusc: nuscenes-devkit NuScenes instance (real path).
        splits_json: path to splits.json from build_splits().
    """
    if mock:
        results = _mock_metrics(retrain_c1=retrain_c1_for_mock, split=split)
    else:
        from .train_c1 import evaluate_c1
        from .train_c2 import evaluate_c2
        from .c3_idm import evaluate_c3

        c1 = evaluate_c1(c1_ckpt, split, splits_json=splits_json, nusc=nusc)
        c2_iso = evaluate_c2(c2_ckpt, split, mode="isolated",
                             splits_json=splits_json, nusc=nusc)
        c2_pipe = evaluate_c2(c2_ckpt, split, mode="pipeline",
                              splits_json=splits_json, nusc=nusc,
                              c1_descriptor_path=c1_ckpt)
        c3_iso = evaluate_c3(c2_checkpoint=c2_ckpt, split=split, mode="isolated",
                             splits_json=splits_json, nusc=nusc)
        c3_pipe = evaluate_c3(
            c2_checkpoint=c2_ckpt, split=split, mode="pipeline",
            c1_checkpoint=c1_ckpt,
            splits_json=splits_json, nusc=nusc,
        )

        results = {
            "c1_mAP":                 c1["mAP"],
            "c2_iso_minADE":          c2_iso["minADE_5"],
            "c2_pipe_minADE":         c2_pipe["minADE_5"],
            "c3_iso_L2":              c3_iso["L2@3s"],
            "c3_pipe_L2":             c3_pipe["L2@3s"],
            "c3_pipe_collision_rate": c3_pipe["collision_rate"],
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Offline self-test
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        baseline = run_all_measurements(
            c1_ckpt="(mock)", c2_ckpt="(mock)",
            split="singapore_val",
            out_path=f"{d}/baseline.json",
            mock=True, retrain_c1_for_mock=False,
        )
        after = run_all_measurements(
            c1_ckpt="(mock)", c2_ckpt="(mock)",
            split="singapore_val",
            out_path=f"{d}/after.json",
            mock=True, retrain_c1_for_mock=True,
        )

    assert after["c1_mAP"] > baseline["c1_mAP"], "C1 should improve"
    assert after["c3_iso_L2"] == baseline["c3_iso_L2"], "C3 iso must be unchanged"
    assert after["c3_pipe_L2"] > baseline["c3_pipe_L2"], "C3 pipe should worsen"
    assert after["c2_iso_minADE"] == baseline["c2_iso_minADE"]
    assert after["c2_pipe_minADE"] > baseline["c2_pipe_minADE"]

    print("baseline:", baseline)
    print("after:", after)
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
