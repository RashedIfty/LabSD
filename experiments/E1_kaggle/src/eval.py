"""Five-measurement harness producing Table I rows."""

import json
from pathlib import Path

from .train_c1 import evaluate_c1
from .train_c2 import evaluate_c2
from .c3_idm import evaluate_c3


def run_all_measurements(
    c1_ckpt: str,
    c2_ckpt: str,
    split: str,
    out_path: str,
) -> dict:
    """
    Compute the five values from Table I of the Meeting 3 report:
        1) C1 mAP
        2) C2 isolated minADE
        3) C2 pipeline minADE
        4) C3 isolated L2
        5) C3 pipeline L2
    """
    # TODO: call each evaluator with correct mode
    results = {
        "c1_mAP":         None,
        "c2_iso_minADE":  None,
        "c2_pipe_minADE": None,
        "c3_iso_L2":      None,
        "c3_pipe_L2":     None,
        "c3_pipe_collision_rate": None,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_all_measurements(
        c1_ckpt="/kaggle/working/c1_boston.pth",
        c2_ckpt="/kaggle/working/c2_boston.pth",
        split="singapore_val",
        out_path="/kaggle/working/baseline.json",
    )
