"""Phase-2b: evaluate the full CARA admission rule on the campaign matrix.

SENSEI_FEEDBACK_PLAN.md §4.2.  The Phase-2a drift gate (drift_gate.py) is CARA's
cheap front-end screen; this module turns its per-update drift scores into an
admit / hold decision and asks whether that decision matches the "good/bad"
ground truth of each retraining update, where ground truth is the measured
terminal metric Delta3 (pipeline planner L2 change) and, as a variant, a
collision-rate-aware version.

Everything here is post-hoc analysis over two artifacts we already have:
  * results/campaign_v30.json   (per-config Delta3 + collision rate)
  * results/gate_eval_v7.json   (per-config drift + cascade_risk)
No model, no GPU.  Run: python -m labsd.cara_eval  (from src/), or import and
call evaluate_cara(campaign_path, gate_path).

Ground-truth convention (Delta3 = after_L2 - baseline_L2):
  * Delta3 > 0  -> the update HURT the planner  -> "bad" -> we WANT to hold it.
  * Delta3 <= 0 -> the update helped/neutral    -> "good" -> we WANT to admit it.

A gate that "holds" an update (cascade_risk = True, i.e. drift > tau) is making a
positive prediction of "bad".  So, treating "bad update" as the positive class:
  TP = held  a bad update      (drift>tau AND Delta3>0)   correct catch
  FP = held  a good update     (drift>tau AND Delta3<=0)  needless hold
  FN = admitted a bad update   (drift<=tau AND Delta3>0)  missed regression
  TN = admitted a good update  (drift<=tau AND Delta3<=0) correct pass
"""
from __future__ import annotations

import json
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load(campaign_path: str, gate_path: str):
    camp = json.loads(Path(campaign_path).read_text())
    gate = json.loads(Path(gate_path).read_text())
    d3 = {r["tag"]: r["Delta3"] for r in camp["rows"]}
    coll = {r["tag"]: r.get("c3_pipe_collision_rate", 0.0) for r in camp["rows"]}
    drift = {r["tag"]: r["drift"] for r in gate["rows"]}
    tags = [r["tag"] for r in camp["rows"] if r["tag"] in drift]
    return tags, d3, coll, drift


# ---------------------------------------------------------------------------
# confusion matrix at a fixed threshold
# ---------------------------------------------------------------------------

def confusion_at_tau(tags, d3, drift, tau: float, bad=lambda x: x > 0.0):
    """Confusion of (hold if drift>tau) against (bad if Delta3>0)."""
    tp = fp = fn = tn = 0
    for t in tags:
        held = drift[t] > tau           # gate's positive prediction ("bad")
        is_bad = bad(d3[t])
        if held and is_bad:   tp += 1
        elif held and not is_bad: fp += 1
        elif not held and is_bad: fn += 1
        else: tn += 1
    return {"tau": tau, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _metrics(cm):
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None           # = catch rate of bad
    spec = tn / (tn + fp) if (tn + fp) else None          # correct-admit rate
    acc = (tp + tn) / n if n else None
    hold_rate = (tp + fp) / n if n else None
    f1 = (2 * prec * rec / (prec + rec)
          if (prec and rec and (prec + rec) > 0) else None)
    return {**cm, "precision": prec, "recall": rec, "specificity": spec,
            "accuracy": acc, "hold_rate": hold_rate, "f1": f1}


# ---------------------------------------------------------------------------
# threshold sweep
# ---------------------------------------------------------------------------

def sweep_tau(tags, d3, drift, bad=lambda x: x > 0.0, n=41):
    lo = min(drift[t] for t in tags)
    hi = max(drift[t] for t in tags)
    pad = 0.02
    taus = [lo - pad + (hi - lo + 2 * pad) * i / (n - 1) for i in range(n)]
    return [_metrics(confusion_at_tau(tags, d3, drift, tau, bad)) for tau in taus]


def best_tau_by_f1(sweep):
    """Best F1, but break ties toward a *discriminating* threshold.

    A hold-everything point (hold_rate=1, specificity=0) can tie on F1 with a
    genuinely selective point; we prefer the selective one (higher specificity,
    then lower hold rate) so the reported operating point actually filters.
    """
    scored = [s for s in sweep if s["f1"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda s: (round(s["f1"], 4),
                                      s["specificity"] or 0.0,
                                      -s["hold_rate"]))


def balanced_operating_point(sweep):
    """The point that best trades catching bad updates against admitting good
    ones: maximise Youden's J = recall + specificity - 1 (ties -> higher F1)."""
    scored = [s for s in sweep
              if s["recall"] is not None and s["specificity"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda s: (s["recall"] + s["specificity"],
                                      s["f1"] or 0.0))


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

def evaluate_cara(campaign_path: str, gate_path: str,
                  tau_default: float = 0.25) -> dict:
    tags, d3, coll, drift = _load(campaign_path, gate_path)

    n = len(tags)
    n_bad = sum(1 for t in tags if d3[t] > 0.0)
    n_good = n - n_bad
    any_collision = any(coll[t] > 0.0 for t in tags)

    # (a) fixed-threshold confusion (the tau the gate shipped with)
    cm_default = _metrics(confusion_at_tau(tags, d3, drift, tau_default))

    # (b) full sweep + best operating points
    sweep = sweep_tau(tags, d3, drift)
    best = best_tau_by_f1(sweep)
    cm_best = _metrics(confusion_at_tau(tags, d3, drift, best["tau"])) if best else None
    bal = balanced_operating_point(sweep)
    cm_bal = _metrics(confusion_at_tau(tags, d3, drift, bal["tau"])) if bal else None

    # rank-separation of the gate: P[drift(bad) > drift(good)] (AUROC).
    bad_dr = [drift[t] for t in tags if d3[t] > 0.0]
    good_dr = [drift[t] for t in tags if d3[t] <= 0.0]
    if bad_dr and good_dr:
        pairs = [(b, g) for b in bad_dr for g in good_dr]
        auc = sum((b > g) + 0.5 * (b == g) for b, g in pairs) / len(pairs)
    else:
        auc = None

    # (c) collision-rate-aware variant.  "bad" = regressed L2 OR introduced a
    #     collision.  With zero collisions everywhere this collapses to (a)/(b);
    #     we compute it anyway and flag the degeneracy.
    bad_coll = lambda t: (d3[t] > 0.0) or (coll[t] > 0.0)
    cm_coll_default = _metrics(
        confusion_at_tau(tags, d3, drift, tau_default,
                         bad=lambda x: x > 0.0))  # placeholder; overwritten below
    # recompute with per-tag collision awareness
    tp = fp = fn = tn = 0
    for t in tags:
        held = drift[t] > tau_default
        is_bad = bad_coll(t)
        if held and is_bad: tp += 1
        elif held and not is_bad: fp += 1
        elif not held and is_bad: fn += 1
        else: tn += 1
    cm_coll_default = _metrics({"tau": tau_default,
                                "TP": tp, "FP": fp, "FN": fn, "TN": tn})

    return {
        "n_updates": n, "n_bad": n_bad, "n_good": n_good,
        "any_collision": any_collision,
        "ground_truth": "Delta3>0 => bad (planner regressed)",
        "auroc_drift_vs_bad": auc,
        "confusion_default_tau": cm_default,
        "best_operating_point": cm_best,
        "balanced_operating_point": cm_bal,
        "collision_aware_default_tau": cm_coll_default,
        "sweep": sweep,
        "per_update": [
            {"tag": t, "drift": drift[t], "Delta3": d3[t],
             "collision_rate": coll[t],
             "bad": d3[t] > 0.0,
             "held_at_0.25": drift[t] > 0.25}
            for t in tags
        ],
    }


def _print_report(res: dict):
    print(f"CARA admission-rule evaluation  (n={res['n_updates']} updates: "
          f"{res['n_bad']} bad / {res['n_good']} good; "
          f"any collision: {res['any_collision']})")
    print(f"ground truth: {res['ground_truth']}")
    auc = res.get("auroc_drift_vs_bad")
    print(f"AUROC (drift ranks bad update above good) = "
          f"{auc:.3f}" if auc is not None else "AUROC: n/a")
    print()

    def show(title, cm):
        if cm is None:
            print(f"{title}: n/a"); return
        print(f"{title}  (tau={cm['tau']:.3f})")
        print(f"   TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}  "
              f"| hold_rate={cm['hold_rate']:.2f} acc={cm['accuracy']:.2f}")
        p = cm['precision']; r = cm['recall']; s = cm['specificity']; f = cm['f1']
        fmt = lambda v: f"{v:.2f}" if v is not None else "n/a"
        print(f"   precision(hold-correct)={fmt(p)}  "
              f"recall(catch-bad)={fmt(r)}  specificity(admit-good)={fmt(s)}  "
              f"F1={fmt(f)}")
        print()

    show("Fixed shipped threshold", res["confusion_default_tau"])
    show("Best operating point (max F1, tie->selective)", res["best_operating_point"])
    show("Balanced operating point (max Youden J)", res["balanced_operating_point"])
    show("Collision-rate-aware rule @0.25", res["collision_aware_default_tau"])

    print("Per-update (sorted by drift):")
    print(f"   {'tag':<16}{'drift':>7}{'Delta3':>9}{'bad':>5}{'held@.25':>10}")
    for u in sorted(res["per_update"], key=lambda x: -x["drift"]):
        print(f"   {u['tag']:<16}{u['drift']:>7.3f}{u['Delta3']:>+9.3f}"
              f"{('Y' if u['bad'] else 'n'):>5}{('HOLD' if u['held_at_0.25'] else 'admit'):>10}")


if __name__ == "__main__":
    import os
    here = Path(__file__).resolve().parent.parent   # experiments/E1_kaggle
    camp = os.environ.get("CAMPAIGN",
                          str(here / "results" / "campaign_v30.json"))
    gate = os.environ.get("GATE",
                          str(here / "results" / "gate_eval_v7.json"))
    res = evaluate_cara(camp, gate)
    _print_report(res)
    out = here / "results" / "cara_eval.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)
