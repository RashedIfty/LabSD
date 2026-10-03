"""Generate Table I and the cascade diagnostic from before/after JSONs.

Table I structure (from Meeting 3 report, §VI):
    | Component       | Metric                  | Before | After | Delta |
    | C1              | mAP Singapore           |        |       |       |
    | C2 (isolated)   | minADE w/ GT detections |        |       |       |
    | C2 (pipeline)   | minADE w/ live C1       |        |       |       |
    | C3 (isolated)   | L2 w/ GT predictions    |        |       |       |
    | C3 (pipeline)   | L2 w/ live C2           |        |       |       |

Cascade diagnostic (also §VI):
    confirmed iff:
        - C3 (pipeline) worsens by more than the noise floor, AND
        - C3 (isolated) is unchanged within the noise floor, AND
        - C1 mAP actually improved (otherwise the retrain didn't take).

If confirmed, also report ρ_1→3 = relative C3 pipeline degradation per
unit relative C1 improvement — the coupling strength used by the SMP.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


METRIC_ROWS = [
    # (Component label, metric label, baseline key, "lower is better"?)
    ("C1",            "mAP Singapore",            "c1_mAP",         False),
    ("C2 (isolated)", "minADE w/ GT detections",  "c2_iso_minADE",  True),
    ("C2 (pipeline)", "minADE w/ live C1",        "c2_pipe_minADE", True),
    ("C3 (isolated)", "L2 w/ GT predictions",     "c3_iso_L2",      True),
    ("C3 (pipeline)", "L2 w/ live C2",            "c3_pipe_L2",     True),
]


def build_table(
    baseline_json: str,
    after_json: str,
    out_csv: str,
) -> pd.DataFrame:
    """Compute deltas and write Table I to CSV. Returns the DataFrame."""
    with open(baseline_json) as f:
        before = json.load(f)
    with open(after_json) as f:
        after = json.load(f)

    rows = []
    for label, metric, key, _lower in METRIC_ROWS:
        b = before.get(key)
        a = after.get(key)
        delta = (a - b) if (a is not None and b is not None) else None
        delta_pct = (100.0 * delta / b) if (delta is not None and b not in (None, 0)) else None
        rows.append({
            "Component": label,
            "Metric": metric,
            "Before": b,
            "After": a,
            "Delta": delta,
            "Delta %": delta_pct,
        })

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def cascade_diagnostic(
    df: pd.DataFrame,
    noise_floor_rel: float = 0.05,
) -> dict:
    """Apply the §VI rule. Returns {'confirmed', 'rho_1_to_3', 'reason'}."""
    by_metric = {row["Metric"]: row for _, row in df.iterrows()}

    c1 = by_metric.get("mAP Singapore")
    c3_iso = by_metric.get("L2 w/ GT predictions")
    c3_pipe = by_metric.get("L2 w/ live C2")

    if c1 is None or c3_iso is None or c3_pipe is None:
        return {"confirmed": False, "rho_1_to_3": None, "reason": "missing rows"}

    # Did C1 actually improve? mAP — higher is better.
    c1_improved = (c1["Delta"] is not None and c1["Delta"] > 0)
    # C3 isolated unchanged?
    c3_iso_stable = (
        c3_iso["Delta %"] is not None
        and abs(c3_iso["Delta %"]) <= 100 * noise_floor_rel
    )
    # C3 pipeline worsened? L2 — higher is worse.
    c3_pipe_worsened = (
        c3_pipe["Delta %"] is not None
        and c3_pipe["Delta %"] > 100 * noise_floor_rel
    )

    confirmed = c1_improved and c3_iso_stable and c3_pipe_worsened

    rho = None
    if confirmed:
        # rel C3 pipe degradation / rel C1 mAP improvement
        rel_c3 = (c3_pipe["Delta"] / c3_pipe["Before"]) if c3_pipe["Before"] else None
        rel_c1 = (c1["Delta"] / c1["Before"]) if c1["Before"] else None
        if rel_c3 is not None and rel_c1 not in (None, 0):
            rho = rel_c3 / rel_c1

    if confirmed:
        reason = "cascade confirmed"
    else:
        bits = []
        if not c1_improved:
            bits.append("C1 did not improve")
        if not c3_iso_stable:
            bits.append("C3 isolated drifted")
        if not c3_pipe_worsened:
            bits.append("C3 pipeline did not worsen")
        reason = "; ".join(bits) or "unknown"

    return {"confirmed": confirmed, "rho_1_to_3": rho, "reason": reason}


# ─────────────────────────────────────────────────────────────────────────────
# Offline self-test — uses eval.py's mock generator end-to-end.
# ─────────────────────────────────────────────────────────────────────────────

def _self_test() -> None:
    import tempfile
    from .eval import run_all_measurements

    with tempfile.TemporaryDirectory() as d:
        run_all_measurements(
            c1_ckpt="(mock)", c2_ckpt="(mock)",
            split="singapore_val",
            out_path=f"{d}/baseline.json",
            mock=True, retrain_c1_for_mock=False,
        )
        run_all_measurements(
            c1_ckpt="(mock)", c2_ckpt="(mock)",
            split="singapore_val",
            out_path=f"{d}/after.json",
            mock=True, retrain_c1_for_mock=True,
        )
        df = build_table(
            baseline_json=f"{d}/baseline.json",
            after_json=f"{d}/after.json",
            out_csv=f"{d}/table_I.csv",
        )

    print(df.to_string(index=False))
    print()
    diag = cascade_diagnostic(df)
    print("diagnostic:", diag)

    assert diag["confirmed"], f"expected cascade-confirmed on mock data: {diag}"
    assert diag["rho_1_to_3"] is not None
    assert diag["rho_1_to_3"] > 0, f"rho should be positive: {diag}"
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
