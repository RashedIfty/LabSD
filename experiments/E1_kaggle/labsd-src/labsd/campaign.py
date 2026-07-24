"""Phase-1 campaign driver: multiple fine-tuning scenarios for the E1 study.

Sensei's Q1 asks whether *different* fine-tuning campaigns produce *different*
entanglements. This module loops the existing single-scenario harness over a
matrix of configs (fine-tune dataset size x epochs x seed), records the full
five-quantity profile before/after for each, and labels the entanglement regime.

It intentionally reuses the existing components unchanged:
  * c1_yolo.build_yolo_dataset / fine_tune_yolo / c1_descriptor_yolo
  * eval.run_all_measurements
  * the cascade quantities delta1, Delta2, Delta3, rho

Design decisions (from SENSEI_FEEDBACK_PLAN.md Phase 0, Kaggle free tier):
  * dataset-size axis is realised by *subsampling images* of the Singapore
    train split (only 3 scenes exist, so image-level is the right granularity);
  * epochs axis uses the existing fine_tune_yolo(epochs=...);
  * seeds are threaded into Ultralytics via the `seed` train arg so mean/std
    can be reported (answers the variance concern behind Q1);
  * every run reuses the SAME frozen Boston-trained C1 as the "before" baseline
    and the SAME C2/C3, so the only thing that varies is the C1 Singapore tune.

Nothing here talks to Kaggle directly; the notebook calls run_campaign().
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Config matrix
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CampaignConfig:
    """One fine-tuning scenario."""
    tag: str            # short unique label, e.g. "size50_ep10_s1"
    frac: float         # fraction of Singapore-train images to keep (0<frac<=1)
    epochs: int
    seed: int
    imgsz: int = 640
    batch: int = 8


def default_matrix() -> list[CampaignConfig]:
    """~18-run minimal-convincing matrix (Phase-0 decision).

    Axis A (generality): dataset size in {full, half, quarter} x 3 seeds.
    Axis B (tune length): epochs in {5, 10, 20} x 3 seeds at full size.
    The (full, 10) cell is shared, so total unique runs = 9 + 9 - 3 = 15
    fine-tunes; we keep them as separate tags for clarity (18 rows, some
    reuse the same cached weights within a session).
    """
    seeds = [1, 2, 3]
    cfgs: list[CampaignConfig] = []
    # Axis A — dataset size (epochs fixed at 10)
    for frac, name in [(1.0, "full"), (0.5, "half"), (0.25, "quarter")]:
        for s in seeds:
            cfgs.append(CampaignConfig(tag=f"{name}_ep10_s{s}", frac=frac,
                                       epochs=10, seed=s))
    # Axis B — epochs (size fixed at full); ep10 already covered above
    for ep in [5, 20]:
        for s in seeds:
            cfgs.append(CampaignConfig(tag=f"full_ep{ep}_s{s}", frac=1.0,
                                       epochs=ep, seed=s))
    return cfgs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset subsampling (image-level, deterministic per seed)
# ─────────────────────────────────────────────────────────────────────────────

def subsample_yolo_dataset(src_dir: str, dst_dir: str, frac: float,
                           seed: int, split_name: str = "train") -> int:
    """Copy a deterministic `frac` of (image,label) pairs from src to dst.

    Keeps the val split untouched (caller copies it whole). Returns the
    number of training pairs kept. frac=1.0 copies everything.
    """
    src_img = Path(src_dir) / "images" / split_name
    src_lbl = Path(src_dir) / "labels" / split_name
    dst_img = Path(dst_dir) / "images" / split_name
    dst_lbl = Path(dst_dir) / "labels" / split_name
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in src_img.glob("*.jpg"))
    if not stems:
        stems = sorted(p.stem for p in src_img.glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(stems)
    keep_n = max(1, round(len(stems) * frac))
    keep = set(stems[:keep_n])

    n = 0
    for stem in keep:
        for ext in (".jpg", ".png"):
            img = src_img / f"{stem}{ext}"
            if img.exists():
                shutil.copy2(img, dst_img / img.name)
                lbl = src_lbl / f"{stem}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, dst_lbl / lbl.name)
                n += 1
                break
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Regime labelling
# ─────────────────────────────────────────────────────────────────────────────

def label_regime(delta1: float, Delta2: float, Delta3: float,
                 eps: float = 1e-6) -> str:
    """Classify the (delta1, Delta2, Delta3) outcome of a fine-tune.

    Conventions (lower error is better, so a NEGATIVE Delta = improvement):
      delta1  = change in C1 aggregate metric (mAP); POSITIVE = better.
      Delta2  = change in C2 pipeline minADE;        NEGATIVE = better.
      Delta3  = change in C3 pipeline L2@3s;         NEGATIVE = better.
    """
    if delta1 is None or Delta2 is None or Delta3 is None:
        return "incomplete"
    c1_up = delta1 > eps
    c1_down = delta1 < -eps
    c2_better = Delta2 < -eps
    c3_worse = Delta3 > eps
    # Order matters: check the most specific/distinctive regime first.
    if c1_up and c3_worse:
        # strict entangled enhancement: C1's own metric up, system output down.
        return "entangled_enhancement"
    if c1_down and c2_better:
        # metric decoupling: aggregate metric fell yet the interface improved
        # (the v22 real run; more specific than plain non-monotonicity).
        return "metric_decoupling"
    if c2_better and c3_worse:
        # non-monotone: an interface improves while the system output degrades.
        return "non_monotone"
    if Delta3 < -eps:
        return "benign"                       # system improved or unchanged
    return "other"


def coupling_rho(delta1: float, Delta3: float, eps: float = 1e-9):
    """rho = Delta3 / delta1. Ambiguous (None) when delta1 ~ 0."""
    if delta1 is None or Delta3 is None or abs(delta1) < eps:
        return None
    return Delta3 / delta1


# ─────────────────────────────────────────────────────────────────────────────
# Campaign loop
# ─────────────────────────────────────────────────────────────────────────────

def run_campaign(
    *,
    nusc,
    splits: dict,
    splits_json: str,
    c1_boston_descriptor: str,   # frozen "before" C1 (Boston-trained)
    c2_ckpt: str,                # frozen C2 (shared)
    baseline_json: str,          # measured once with the Boston C1
    sg_full_dataset_dir: str,    # pre-built full Singapore YOLO dataset
    work_root: str,
    configs: list[CampaignConfig] | None = None,
    nusc_maps=None,
    enable_pkl: bool = False,
) -> dict:
    """Run every config; return {rows:[...], baseline:{...}}.

    For each config: subsample -> fine-tune C1 on Singapore -> measure the
    5-quantity profile -> compute deltas against the shared baseline -> label.
    """
    from .c1_yolo import (fine_tune_yolo, write_yolo_data_yaml,
                          c1_descriptor_yolo)
    from .eval import run_all_measurements

    configs = configs or default_matrix()
    Path(work_root).mkdir(parents=True, exist_ok=True)

    with open(baseline_json) as f:
        base = json.load(f)
    base_c1 = base.get("c1_mAP")
    base_c2 = base.get("c2_pipe_minADE")
    base_c3 = base.get("c3_pipe_L2")

    rows = []
    for cfg in configs:
        run_dir = Path(work_root) / cfg.tag
        data_dir = run_dir / "data"
        # 1) subsample training images; copy val split whole
        n_tr = subsample_yolo_dataset(sg_full_dataset_dir, str(data_dir),
                                      frac=cfg.frac, seed=cfg.seed,
                                      split_name="train")
        _copy_split_whole(sg_full_dataset_dir, str(data_dir), "val")
        data_yaml = write_yolo_data_yaml(str(data_dir))

        # 2) fine-tune C1 on Singapore from the SAME Boston weights
        boston_weights = _boston_weights_from_descriptor(c1_boston_descriptor)
        best = fine_tune_yolo(
            data_yaml=data_yaml, out_dir=str(run_dir),
            base_weights=boston_weights, epochs=cfg.epochs,
            imgsz=cfg.imgsz, batch=cfg.batch,
            name=f"c1_{cfg.tag}", seed=cfg.seed,
        )
        c1_after = c1_descriptor_yolo(str(run_dir), best,
                                      label=f"sg_{cfg.tag}",
                                      val_data_yaml=data_yaml)

        # 3) measure the after-profile (C2/C3 frozen)
        after = run_all_measurements(
            c1_ckpt=c1_after, c2_ckpt=c2_ckpt, split="singapore_val",
            out_path=str(run_dir / "after.json"),
            nusc=nusc, splits_json=splits_json,
            nusc_maps=nusc_maps, enable_pkl=enable_pkl,
        )

        # 3b) pure cascade metric at the planner: how far the PLAN moves when
        #     C1 is swapped from the frozen Boston baseline to this fine-tune.
        #     This has no human-mismatch offset, so it isolates the cascade
        #     contribution at C3 (0 = plan unchanged; larger = bigger cascade).
        from .c3_idm import plan_shift_between
        shift = plan_shift_between(
            before_c1=c1_boston_descriptor, after_c1=c1_after,
            split="singapore_val", splits_json=splits_json, nusc=nusc,
        )
        plan_shift3 = shift.get("plan_shift@3s")

        # 4) deltas vs shared baseline + regime label
        d1 = _sub(after.get("c1_mAP"), base_c1)          # POSITIVE = C1 better
        D2 = _sub(after.get("c2_pipe_minADE"), base_c2)  # NEGATIVE = C2 better
        D3 = _sub(after.get("c3_pipe_L2"), base_c3)      # NEGATIVE = C3 better
        row = {
            **asdict(cfg),
            "n_train": n_tr,
            "c1_mAP": after.get("c1_mAP"),
            "c2_pipe_minADE": after.get("c2_pipe_minADE"),
            "c3_pipe_L2": after.get("c3_pipe_L2"),
            "c3_pipe_collision_rate": after.get("c3_pipe_collision_rate"),
            "delta1": d1, "Delta2": D2, "Delta3": D3,
            "plan_shift3": plan_shift3,   # pure cascade signal at the planner
            "rho": coupling_rho(d1, D3),
            "regime": label_regime(d1, D2, D3),
        }
        rows.append(row)
        (run_dir / "row.json").write_text(json.dumps(row, indent=2))
        (run_dir / "plan_shift.json").write_text(json.dumps(shift, indent=2))

    out = {"baseline": {"c1_mAP": base_c1, "c2_pipe_minADE": base_c2,
                        "c3_pipe_L2": base_c3},
           "rows": rows}
    Path(work_root, "campaign_results.json").write_text(json.dumps(out, indent=2))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase-2a: retrospective drift-gate evaluation on a finished campaign
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gate_on_campaign(
    *,
    nusc,
    splits: dict,
    campaign_result: dict,        # output of run_campaign()
    c1_boston_weights: str,       # incumbent detector weights (before)
    work_root: str,               # where each config's run dir lives
    tau: float = 0.25,
    split: str = "singapore_val",
) -> dict:
    """For each config, compute the drift-gate score of the retrained detector
    vs. the incumbent, and pair it with that config's measured plan shift.

    This lets us ask (offline, no retraining): does a high drift flag the
    campaigns that actually moved the plan most? Returns rows with
    {tag, drift, cascade_risk, plan_shift3} plus a rank correlation between
    drift and plan_shift3.
    """
    from .c1_yolo import dump_detections, load_yolo_descriptor
    from .drift_gate import gate_decision

    scenes = splits.get(split, [])
    old_dets = dump_detections(c1_boston_weights, scenes, nusc)

    rows = []
    for r in campaign_result["rows"]:
        tag = r["tag"]
        descr = Path(work_root) / tag / "c1_descriptor.json"
        new_weights, _ = load_yolo_descriptor(str(descr))
        new_dets = dump_detections(new_weights, scenes, nusc)
        g = gate_decision(old_dets, new_dets, tau=tau)
        rows.append({
            "tag": tag,
            "drift": g["drift"],
            "per_feature": g["per_feature"],
            "cascade_risk": g["cascade_risk"],
            "plan_shift3": r.get("plan_shift3"),
            "Delta3": r.get("Delta3"),
        })

    corr = _spearman([x["drift"] for x in rows],
                     [x["plan_shift3"] for x in rows])
    out = {"tau": tau, "rows": rows, "spearman_drift_vs_planshift": corr}
    Path(work_root, "gate_eval.json").write_text(json.dumps(out, indent=2))
    return out


def _spearman(a: list, b: list):
    """Spearman rank correlation between two equal-length numeric lists."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    xa = [p[0] for p in pairs]
    xb = [p[1] for p in pairs]

    def _ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    ra, rb = _ranks(xa), _ranks(xb)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    da = math.sqrt(sum((ra[i] - mean_a) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mean_b) ** 2 for i in range(n)))
    if da == 0 or db == 0:
        return None
    return num / (da * db)


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sub(a, b):
    if a is None or b is None:
        return None
    return a - b


def _copy_split_whole(src_dir: str, dst_dir: str, split_name: str) -> None:
    for kind in ("images", "labels"):
        s = Path(src_dir) / kind / split_name
        d = Path(dst_dir) / kind / split_name
        d.mkdir(parents=True, exist_ok=True)
        if s.exists():
            for p in s.iterdir():
                shutil.copy2(p, d / p.name)


def _boston_weights_from_descriptor(descriptor_path: str) -> str:
    d = json.loads(Path(descriptor_path).read_text())
    return d["weights_path"]


# ─────────────────────────────────────────────────────────────────────────────
# Class-balanced fine-tune: the run that chases delta1 > 0
# ─────────────────────────────────────────────────────────────────────────────

def run_class_balanced(
    *,
    nusc,
    splits: dict,
    splits_json: str,
    c1_boston_descriptor: str,
    c2_ckpt: str,
    baseline_json: str,
    sg_full_dataset_dir: str,
    work_root: str,
    epochs_list: tuple[int, ...] = (10, 20),
    seeds: tuple[int, ...] = (1, 2, 3),
    powers: tuple[float, ...] = (0.5, 1.0),
    max_repeat: int = 6,
    nusc_maps=None,
    enable_pkl: bool = False,
) -> dict:
    """Fine-tune C1 on a *class-balanced* Singapore train split.

    Every campaign run so far lost aggregate mAP (delta1 < 0) because the
    119-image fine-tune set is dominated by `car`. Balancing the class mix by
    oversampling rare-class images is the Meeting-4 "next step" intended to push
    delta1 positive. If delta1 > 0 while the terminal metric still degrades
    (Delta3 > 0), that is the STRICT entangled-enhancement condition.

    The val split is copied verbatim, so mAP remains comparable to the
    unbalanced campaign runs.
    """
    from .c1_yolo import (fine_tune_yolo, write_yolo_data_yaml,
                          c1_descriptor_yolo)
    from .class_balance import build_class_balanced_split, copy_split
    from .eval import run_all_measurements
    from .c3_idm import plan_shift_between

    Path(work_root).mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(baseline_json).read_text())
    base_c1 = base.get("c1_mAP")
    base_c2 = base.get("c2_pipe_minADE")
    base_c3 = base.get("c3_pipe_L2")
    boston_weights = _boston_weights_from_descriptor(c1_boston_descriptor)

    rows = []
    balance_reports = {}
    for power in powers:
        # one balanced dataset per `power` setting, reused across epochs/seeds
        data_dir = Path(work_root) / f"data_p{str(power).replace('.', '')}"
        report = build_class_balanced_split(
            sg_full_dataset_dir, str(data_dir), split="train",
            max_repeat=max_repeat, power=power,
        )
        copy_split(sg_full_dataset_dir, str(data_dir), "val")
        data_yaml = write_yolo_data_yaml(str(data_dir))
        balance_reports[f"power_{power}"] = report
        print(f"[balanced p={power}] {report.get('src_images')} -> "
              f"{report.get('balanced_images')} imgs | imbalance "
              f"{report.get('imbalance_before')} -> {report.get('imbalance_after')}")

        for ep in epochs_list:
            for sd in seeds:
                tag = f"bal_p{str(power).replace('.', '')}_ep{ep}_s{sd}"
                run_dir = Path(work_root) / tag
                best = fine_tune_yolo(
                    data_yaml=data_yaml, out_dir=str(run_dir),
                    base_weights=boston_weights, epochs=ep,
                    imgsz=640, batch=8, name=f"c1_{tag}", seed=sd,
                )
                c1_after = c1_descriptor_yolo(str(run_dir), best,
                                              label=f"sg_{tag}",
                                              val_data_yaml=data_yaml)
                after = run_all_measurements(
                    c1_ckpt=c1_after, c2_ckpt=c2_ckpt, split="singapore_val",
                    out_path=str(run_dir / "after.json"),
                    nusc=nusc, splits_json=splits_json,
                    nusc_maps=nusc_maps, enable_pkl=enable_pkl,
                )
                shift = plan_shift_between(
                    before_c1=c1_boston_descriptor, after_c1=c1_after,
                    split="singapore_val", splits_json=splits_json, nusc=nusc,
                )
                d1 = _sub(after.get("c1_mAP"), base_c1)
                D2 = _sub(after.get("c2_pipe_minADE"), base_c2)
                D3 = _sub(after.get("c3_pipe_L2"), base_c3)
                row = {
                    "tag": tag, "power": power, "epochs": ep, "seed": sd,
                    "n_train": report.get("balanced_images"),
                    "c1_mAP": after.get("c1_mAP"),
                    "c2_pipe_minADE": after.get("c2_pipe_minADE"),
                    "c3_pipe_L2": after.get("c3_pipe_L2"),
                    "c3_pipe_collision_rate": after.get("c3_pipe_collision_rate"),
                    "delta1": d1, "Delta2": D2, "Delta3": D3,
                    "plan_shift3": shift.get("plan_shift@3s"),
                    "rho": coupling_rho(d1, D3),
                    "regime": label_regime(d1, D2, D3),
                    # the headline test: C1 improves AND system degrades
                    "strict_entangled_enhancement": bool(
                        d1 is not None and D3 is not None and d1 > 0 and D3 > 0
                    ),
                }
                rows.append(row)
                (run_dir / "row.json").write_text(json.dumps(row, indent=2))
                print(f"  {tag:<22} d1={d1:+.4f} D2={D2:+.3f} D3={D3:+.3f} "
                      f"EE={row['strict_entangled_enhancement']}")

    n_pos = sum(1 for r in rows if (r['delta1'] or 0) > 0)
    n_ee = sum(1 for r in rows if r['strict_entangled_enhancement'])
    out = {
        "baseline": {"c1_mAP": base_c1, "c2_pipe_minADE": base_c2,
                     "c3_pipe_L2": base_c3},
        "balance_reports": balance_reports,
        "rows": rows,
        "n_delta1_positive": n_pos,
        "n_strict_entangled_enhancement": n_ee,
    }
    Path(work_root, "class_balanced_results.json").write_text(
        json.dumps(out, indent=2))
    print(f"\n[class-balanced] delta1>0 in {n_pos}/{len(rows)} runs; "
          f"STRICT entangled enhancement in {n_ee}/{len(rows)}")
    return out
