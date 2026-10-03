"""Stage 1b: Stage 1 with the overfitting fixes and two C3 variants.

Changes from Stage 1: Boston validation split (every 7th scene) with early
stopping on it, mirror + rotation augmentation, and two C3 variants trained on
the same C2: "noego" (report design) and "ego" (+ ego past path, 1.5 s).
The isolated setting does not depend on C1 and is trained once.
Baselines are computed on the same test anchors and stored with the results.

python -m mlpipe.run_stage1b --c1-seed 0 [--smoke]
"""
import argparse, json, time
from pathlib import Path

import numpy as np

from . import data as D
from .evaluate import c2_metrics, c3_metrics, object_use, c2_output_change
from .models import C2AutoBot, C3PlanT
from .train import (train_c2, predict_c2, train_c3, plan_c3, c3_objects_from_gt,
                    c3_objects_from_c2, save_model, load_model)

ROOT = Path(__file__).resolve().parent.parent
C3_VARIANTS = {"noego": False, "ego": True}


def baselines(train_s, test_s):
    """C2: object stays where it is. C3: mean Boston path per command (no inputs at all)."""
    stay = [np.linalg.norm(s.fut[q][s.fut_mask[q]] - s.cur_xy[q], axis=-1).mean()
            for s in test_s for q in range(len(s.current)) if s.fut_mask[q].any()]
    E = np.array([s.ego_fut for s in train_s]); c = np.array([s.command for s in train_s])
    mean = {k: E[c == k].mean(0) for k in range(3) if (c == k).any()}
    plans = np.array([mean.get(s.command, E.mean(0)) for s in test_s])
    return {"c2_stay_still_ADE": float(np.mean(stay)), "c3_mean_path": c3_metrics(test_s, plans)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    X = a.c1_seed
    tag = "smoke1b" if a.smoke else "stage1b"
    mdir = ROOT / "models" / tag; rdir = ROOT / "results" / tag
    mdir.mkdir(parents=True, exist_ok=True); rdir.mkdir(parents=True, exist_ok=True)
    logf = open(rdir / f"p1_c1s{X}_seed{a.seed}.log", "a")
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    tr = D.load_export(ROOT / "data" / "stage1" / "boston_c23.json")
    te = D.load_export(ROOT / "data" / "stage1" / "singapore_test.json")
    tr_scenes, va_scenes = D.split_train_val(tr)
    fte = None
    ep2, ep3 = 40, 60
    if a.smoke:
        tr_scenes = set(sorted(tr_scenes)[:6]); va_scenes = set(sorted(va_scenes)[:2])
        fte = {s["scene"] for s in te["scenes"][:3]}; ep2, ep3 = 2, 2
    old, new, other = f"c1_old_s{X}", f"c1_new_s{X}", f"c1_old_s{1 - X}"
    settings = {"isolated": "gt", "old_pipeline": f"det:{old}", "new_pipeline": f"det:{new}"}
    man = mdir / "manifest.json"

    test = {src: D.build_samples(te, src, fte) for src in ["gt", f"det:{old}", f"det:{new}", f"det:{other}"]}
    anchors = [s.anchor for s in test["gt"]]
    assert all([s.anchor for s in v] == anchors for v in test.values())
    log(f"P1 c1-seed {X}: {len(anchors)} test anchors, {len(tr_scenes)} train / {len(va_scenes)} val Boston scenes")
    out = {"c1_seed": X, "c2c3_seed": a.seed, "n_test_anchors": len(anchors), "settings": {},
           "baselines": baselines(D.build_samples(tr, "gt", tr_scenes), test["gt"])}
    log("baselines: " + json.dumps(out["baselines"]["c2_stay_still_ADE"]) + " / C3 mean path L2@3s "
        + f"{out['baselines']['c3_mean_path']['L2@3s']:.3f}")

    for name, src in settings.items():
        stem = f"p1_{name}" + ("" if name == "isolated" else f"_c1s{X}") + f"_seed{a.seed}"
        f2 = mdir / f"{stem}_c2.pt"
        trS, vaS = D.build_samples(tr, src, tr_scenes), D.build_samples(tr, src, va_scenes)
        info = {"pipeline": "P1", "setting": name, "trained_on": f"boston_c23 train scenes / {src}",
                "validation": "boston_c23 every 7th scene", "c1_seed": None if name == "isolated" else X,
                "seed": a.seed, "augmentation": "mirror + rotation 10 deg", "early_stopping": True}
        if f2.exists():
            c2 = load_model(C2AutoBot, f2); log(f"{name}: loaded C2")
        else:
            t = time.time(); log(f"{name}: training C2 on {src}: {len(trS)} train / {len(vaS)} val anchors")
            c2 = train_c2(trS, vaS, seed=a.seed, epochs=ep2, log=log)
            save_model(c2, f2, man, f2.stem, {**info, "module": "C2 AutoBot-Ego style", "best_val_minADE": c2.best_val})
            log(f"{name}: C2 done in {time.time() - t:.0f}s, best val minADE {c2.best_val:.3f}")
        if src == "gt":
            o_tr = [c3_objects_from_gt(s) for s in trS]; o_va = [c3_objects_from_gt(s) for s in vaS]
        else:
            o_tr = [c3_objects_from_c2(s, p) for s, p in zip(trS, predict_c2(c2, trS))]
            o_va = [c3_objects_from_c2(s, p) for s, p in zip(vaS, predict_c2(c2, vaS))]

        p_iso = predict_c2(c2, test["gt"]); objs_iso = [c3_objects_from_gt(s) for s in test["gt"]]
        evals = [old, new, other] if name != "new_pipeline" else [new]
        preds = {c1: predict_c2(c2, test[f"det:{c1}"]) for c1 in evals}
        res = {"c2_iso": c2_metrics(test["gt"], p_iso),
               "c2_pipe": {c1: c2_metrics(test[f"det:{c1}"], preds[c1]) for c1 in evals},
               "c2_stay_still_on_scored": {c1: float(np.mean([
                   np.linalg.norm(s.fut[q][s.fut_mask[q]] - s.cur_xy[q], axis=-1).mean()
                   for s in test[f"det:{c1}"] for q in range(len(s.current)) if s.fut_mask[q].any()]))
                   for c1 in evals},
               "c3": {}}
        if name != "new_pipeline":
            res["delta2_minADE"] = res["c2_pipe"][new]["minADE"] - res["c2_pipe"][old]["minADE"]
            res["seed_noise_delta2_minADE"] = res["c2_pipe"][other]["minADE"] - res["c2_pipe"][old]["minADE"]
            res["h2_c2_change_update"] = c2_output_change(test[f"det:{old}"], preds[old], test[f"det:{new}"], preds[new])
            res["h2_c2_change_seed"] = c2_output_change(test[f"det:{old}"], preds[old], test[f"det:{other}"], preds[other])

        for var, use_ego in C3_VARIANTS.items():
            f3 = mdir / f"{stem}_c3_{var}.pt"
            if f3.exists():
                c3 = load_model(C3PlanT, f3, use_ego=use_ego); log(f"{name}/{var}: loaded C3")
            else:
                t = time.time(); log(f"{name}/{var}: training C3")
                c3 = train_c3(o_tr, trS, o_va, vaS, seed=a.seed, epochs=ep3, use_ego=use_ego, log=log)
                save_model(c3, f3, man, f3.stem, {**info, "module": f"C3 PlanT style ({var})",
                                                   "use_ego": use_ego, "best_val_L2@3s": c3.best_val})
                log(f"{name}/{var}: C3 done in {time.time() - t:.0f}s, best val L2@3s {c3.best_val:.3f}")
            plan_iso = plan_c3(c3, objs_iso, test["gt"])
            r = {"c3_iso": c3_metrics(test["gt"], plan_iso),
                 "object_use_iso_m": object_use(plan_iso, plan_c3(c3, objs_iso, test["gt"], drop_objects=True)),
                 "pipe": {}}
            for c1 in evals:
                S = test[f"det:{c1}"]
                objs = [c3_objects_from_c2(s, p) for s, p in zip(S, preds[c1])]
                plans = plan_c3(c3, objs, S)
                r["pipe"][c1] = {"c3": c3_metrics(S, plans),
                                 "object_use_m": object_use(plans, plan_c3(c3, objs, S, drop_objects=True))}
            r["isolation_exact"] = bool(np.array_equal(plan_c3(c3, objs_iso, test["gt"]), plan_iso))
            if name != "new_pipeline":
                r["delta3_L2@3s"] = r["pipe"][new]["c3"]["L2@3s"] - r["pipe"][old]["c3"]["L2@3s"]
                r["seed_noise_delta3_L2@3s"] = r["pipe"][other]["c3"]["L2@3s"] - r["pipe"][old]["c3"]["L2@3s"]
            res["c3"][var] = r
            log(f"{name}/{var}: iso L2@3s {r['c3_iso']['L2@3s']:.3f} | " + " ".join(
                f"{c1} {v['c3']['L2@3s']:.3f}" for c1, v in r["pipe"].items())
                + (f" | delta3 {r['delta3_L2@3s']:+.3f} noise {r['seed_noise_delta3_L2@3s']:+.3f}" if "delta3_L2@3s" in r else "")
                + f" | obj-use {r['object_use_iso_m']:.2f} | exact {r['isolation_exact']}")
        log(f"{name}: C2 iso minADE {res['c2_iso']['minADE']:.3f} | " + " ".join(
            f"{c1} {v['minADE']:.3f}" for c1, v in res["c2_pipe"].items())
            + (f" | delta2 {res['delta2_minADE']:+.3f} noise {res['seed_noise_delta2_minADE']:+.3f}" if "delta2_minADE" in res else ""))
        out["settings"][name] = res
    json.dump(out, open(rdir / f"p1_c1s{X}_seed{a.seed}.json", "w"), indent=1)
    log("DONE -> " + str(rdir / f"p1_c1s{X}_seed{a.seed}.json"))


if __name__ == "__main__":
    main()
