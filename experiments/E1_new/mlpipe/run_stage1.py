"""Stage 1: train C2/C3 in three ways for pipeline P1 and test H1 (one C1 seed).

python -m mlpipe.run_stage1 --c1-seed 0 [--smoke]
Models are saved to models/stage1/ and reused if present (never retrained).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .evaluate import c2_metrics, c3_metrics, object_use, c2_output_change
from .models import C2AutoBot, C3PlanT
from .train import (train_c2, predict_c2, train_c3, plan_c3, c3_objects_from_gt,
                    c3_objects_from_c2, save_model, load_model)

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="C2/C3 training seed")
    ap.add_argument("--smoke", action="store_true", help="5 scenes, few epochs, separate output dir")
    a = ap.parse_args()
    X = a.c1_seed
    tag = "smoke" if a.smoke else "stage1"
    mdir = ROOT / "models" / tag; rdir = ROOT / "results" / tag
    mdir.mkdir(parents=True, exist_ok=True); rdir.mkdir(parents=True, exist_ok=True)
    logf = open(rdir / f"p1_c1s{X}_seed{a.seed}.log", "a")
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    tr = D.load_export(ROOT / "data" / "stage1" / "boston_c23.json")
    te = D.load_export(ROOT / "data" / "stage1" / "singapore_test.json")
    ftr = fte = None
    ep2, ep3 = 40, 60
    if a.smoke:
        ftr = {s["scene"] for s in tr["scenes"][:5]}; fte = {s["scene"] for s in te["scenes"][:3]}
        ep2, ep3 = 2, 2
    old, new, other = f"c1_old_s{X}", f"c1_new_s{X}", f"c1_old_s{1 - X}"
    settings = {"isolated": "gt", "old_pipeline": f"det:{old}", "new_pipeline": f"det:{new}"}
    man = mdir / "manifest.json"

    test = {src: D.build_samples(te, src, fte) for src in ["gt", f"det:{old}", f"det:{new}", f"det:{other}"]}
    anchors = [s.anchor for s in test["gt"]]
    assert all([s.anchor for s in v] == anchors for v in test.values()), "test anchors must match"
    log(f"P1 c1-seed {X}: {len(anchors)} test anchors")

    out = {"c1_seed": X, "c2c3_seed": a.seed, "n_test_anchors": len(anchors), "settings": {}}
    for name, src in settings.items():
        f2, f3 = mdir / f"p1_{name}_c1s{X}_seed{a.seed}_c2.pt", mdir / f"p1_{name}_c1s{X}_seed{a.seed}_c3.pt"
        train_s = D.build_samples(tr, src, ftr)
        if f2.exists() and f3.exists():
            c2, c3 = load_model(C2AutoBot, f2), load_model(C3PlanT, f3); log(f"{name}: loaded saved models")
        else:
            t = time.time(); log(f"{name}: training C2 on {src}, {len(train_s)} anchors")
            c2 = train_c2(train_s, seed=a.seed, epochs=ep2, log=log)
            objs = ([c3_objects_from_gt(s) for s in train_s] if src == "gt"
                    else [c3_objects_from_c2(s, p) for s, p in zip(train_s, predict_c2(c2, train_s))])
            c3 = train_c3(objs, train_s, seed=a.seed, epochs=ep3, log=log)
            info = {"pipeline": "P1", "setting": name, "trained_on": f"boston_c23 / {src}",
                    "c1_seed": X, "seed": a.seed, "epochs": [ep2, ep3]}
            save_model(c2, f2, man, f2.stem, {**info, "module": "C2 AutoBot-Ego style"})
            save_model(c3, f3, man, f3.stem, {**info, "module": "C3 PlanT style"})
            log(f"{name}: trained and saved in {time.time() - t:.0f}s")

        # Isolated measurement (GT input): must not depend on C1
        p_iso = predict_c2(c2, test["gt"]); objs_iso = [c3_objects_from_gt(s) for s in test["gt"]]
        plan_iso = plan_c3(c3, objs_iso, test["gt"])
        res = {"c2_iso": c2_metrics(test["gt"], p_iso), "c3_iso": c3_metrics(test["gt"], plan_iso),
               "object_use_iso_m": object_use(plan_iso, plan_c3(c3, objs_iso, test["gt"], drop_objects=True)),
               "pipe": {}}
        evals = [old, new, other] if name != "new_pipeline" else [new]
        preds = {}
        for c1 in evals:
            S = test[f"det:{c1}"]; P = predict_c2(c2, S); preds[c1] = P
            objs = [c3_objects_from_c2(s, p) for s, p in zip(S, P)]
            plans = plan_c3(c3, objs, S)
            res["pipe"][c1] = {"c2": c2_metrics(S, P), "c3": c3_metrics(S, plans),
                               "object_use_m": object_use(plans, plan_c3(c3, objs, S, drop_objects=True))}
        # Isolation and determinism check: repeat the isolated measurement after the pipeline runs
        again = plan_c3(c3, objs_iso, test["gt"])
        res["isolation_exact"] = bool(np.array_equal(again, plan_iso))
        if name != "new_pipeline":
            po, pn = res["pipe"][old], res["pipe"][new]
            res["delta2_minADE"] = pn["c2"]["minADE"] - po["c2"]["minADE"]
            res["delta3_L2@3s"] = pn["c3"]["L2@3s"] - po["c3"]["L2@3s"]
            res["seed_noise_delta3_L2@3s"] = res["pipe"][other]["c3"]["L2@3s"] - po["c3"]["L2@3s"]
            res["seed_noise_delta2_minADE"] = res["pipe"][other]["c2"]["minADE"] - po["c2"]["minADE"]
            res["h2_c2_change_update"] = c2_output_change(test[f"det:{old}"], preds[old], test[f"det:{new}"], preds[new])
            res["h2_c2_change_seed"] = c2_output_change(test[f"det:{old}"], preds[old], test[f"det:{other}"], preds[other])
        out["settings"][name] = res
        log(f"{name}: " + json.dumps({k: v for k, v in res.items() if k.startswith(('delta', 'seed', 'isolation', 'object'))}))
    json.dump(out, open(rdir / f"p1_c1s{X}_seed{a.seed}.json", "w"), indent=1)
    log("DONE -> " + str(rdir / f"p1_c1s{X}_seed{a.seed}.json"))


if __name__ == "__main__":
    main()
