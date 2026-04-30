"""Generate Table I from before/after measurement JSONs."""

import json
from pathlib import Path

import pandas as pd


def build_table(baseline_json: str, after_json: str, out_csv: str) -> pd.DataFrame:
    """Compute deltas and the cascade diagnostic."""
    with open(baseline_json) as f:
        before = json.load(f)
    with open(after_json) as f:
        after = json.load(f)

    rows = [
        ("C1",            "mAP Singapore",
         before["c1_mAP"], after["c1_mAP"]),
        ("C2 (isolated)", "minADE w/ GT detections",
         before["c2_iso_minADE"], after["c2_iso_minADE"]),
        ("C2 (pipeline)", "minADE w/ live C1",
         before["c2_pipe_minADE"], after["c2_pipe_minADE"]),
        ("C3 (isolated)", "L2 w/ GT predictions",
         before["c3_iso_L2"], after["c3_iso_L2"]),
        ("C3 (pipeline)", "L2 w/ live C2",
         before["c3_pipe_L2"], after["c3_pipe_L2"]),
    ]
    df = pd.DataFrame(rows, columns=["Component", "Metric", "Before", "After"])
    # TODO: compute Delta and Delta% safely (handle None)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def cascade_diagnostic(df: pd.DataFrame, noise_floor: float = 0.05) -> dict:
    """Return {'confirmed': bool, 'rho_1_to_3': float | None}."""
    # TODO: implement the rule from §VI-B of Meeting 3 report
    return {"confirmed": False, "rho_1_to_3": None}


if __name__ == "__main__":
    df = build_table(
        baseline_json="/kaggle/working/baseline.json",
        after_json="/kaggle/working/after_retrain.json",
        out_csv="/kaggle/working/table_I.csv",
    )
    print(df.to_markdown(index=False))
    print(cascade_diagnostic(df))
