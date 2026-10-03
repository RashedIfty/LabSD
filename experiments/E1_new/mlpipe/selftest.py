"""Self-tests on a synthetic export (no nuScenes needed). Run: python -m mlpipe.selftest"""
import math

import numpy as np
import torch

from . import data as D
from .evaluate import c2_metrics, c3_metrics, object_use, c2_output_change
from .train import (train_c2, predict_c2, train_c3, plan_c3, c3_objects_from_gt,
                    c3_objects_from_c2)


def synthetic_export(n_scenes=6, n_frames=14, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    scenes = []
    dets = {"m": {}}
    for sc in range(n_scenes):
        yaw = 0.3 + 0.2 * sc + shift; speed = 4.0 + sc + shift   # m/s
        samples = []
        parked = (20.0 * math.cos(yaw) - 3 * math.sin(yaw), 20.0 * math.sin(yaw) + 3 * math.cos(yaw))
        for f in range(n_frames):
            t = int(f * 0.5e6)
            ex, ey = speed * f * 0.5 * math.cos(yaw), speed * f * 0.5 * math.sin(yaw)
            mover = (ex + 15 * math.cos(yaw) + 2 * f * math.cos(yaw), ey + 15 * math.sin(yaw) + 2 * f * math.sin(yaw))
            gt = [["parked", 0, parked[0], parked[1], yaw, 1.9, 4.5, 1, 50],
                  ["mover", 7, mover[0], mover[1], yaw, 0.6, 0.8, 1, 20]]
            tok = f"s{sc}_{f}"
            samples.append({"token": tok, "t": t, "ego": [ex, ey, yaw], "gt": gt})
            ego = (ex, ey, yaw)
            loc = D.to_local(np.array([[g[2], g[3]] for g in gt]), ego)
            dets["m"][tok] = [[g[1], float(p[0] + rng.normal(0, 0.3)), float(p[1] + rng.normal(0, 0.3)), 0.8]
                              for g, p in zip(gt, loc)]
        scenes.append({"scene": f"sc{sc}", "name": f"sc{sc}", "samples": samples})
    return {"scenes": scenes, "detections": dets}


def main():
    ex = synthetic_export()
    gt_s = D.build_samples(ex, "gt")
    det_s = D.build_samples(ex, "det:m")
    assert len(gt_s) == len(det_s) == 6 * (14 - D.FUT), len(gt_s)

    s = gt_s[3]
    q = [i for i in range(len(s.current)) if s.cur_cls[i] == 0][0]
    assert np.allclose(s.fut[q], s.cur_xy[q][None], atol=1e-6), "parked car must stay still in frame i"
    assert np.allclose(s.cur_xy[q], D.to_local(np.array([[ex["scenes"][0]["samples"][3]["gt"][0][2],
                                                          ex["scenes"][0]["samples"][3]["gt"][0][3]]]),
                                               tuple(ex["scenes"][0]["samples"][3]["ego"]))[0], atol=1e-6)
    speed = 4.0
    assert np.allclose(s.ego_fut[:, 0], speed * 0.5 * np.arange(1, 7), atol=1e-5), s.ego_fut
    assert np.allclose(s.ego_fut[:, 1], 0, atol=1e-5) and s.command == 1
    assert len(s.tokens) == 2 * D.HIST and s.tokens[:, -1].min() == -1.5
    print("transforms, ego path, command, history: OK")

    sd = det_s[3]
    assert sd.fut_mask.all(), "every noisy detection must be assigned to its GT object"
    assert np.allclose(sd.fut[sd.cur_cls == 0][0], s.fut[q], atol=1e-6)
    print("detection -> GT assignment: OK")

    torch.manual_seed(0)
    c2 = train_c2(gt_s, epochs=150, bs=16, lr=1e-3, patience=1000, aug=False, log=lambda *_: None)
    pr = predict_c2(c2, gt_s)
    m2 = c2_metrics(gt_s, pr)
    stay = np.mean([np.linalg.norm(x.fut[i][x.fut_mask[i]] - x.cur_xy[i], axis=-1).mean()
                    for x in gt_s for i in range(len(x.current)) if x.fut_mask[i].any()])
    print(f"C2 overfit: minADE {m2['minADE']:.3f} m vs stay-still {stay:.3f} m")
    assert m2["minADE"] < 0.5 * stay

    # Generalisation: train on 12 scenes, test on 6 unseen scenes with other speeds and headings
    tr_g = D.build_samples(synthetic_export(n_scenes=12, seed=3), "gt")
    te_g = D.build_samples(synthetic_export(n_scenes=6, seed=4, shift=0.5), "gt")
    c2g = train_c2(tr_g, epochs=60, bs=16, lr=1e-3, patience=1000, aug=True, log=lambda *_: None)
    mg = c2_metrics(te_g, predict_c2(c2g, te_g))
    stay_g = np.mean([np.linalg.norm(x.fut[i][x.fut_mask[i]] - x.cur_xy[i], axis=-1).mean()
                      for x in te_g for i in range(len(x.current)) if x.fut_mask[i].any()])
    print(f"C2 on unseen scenes: minADE {mg['minADE']:.3f} m vs stay-still {stay_g:.3f} m")
    assert mg["minADE"] < 0.3 * stay_g, "C2 must learn motion from earlier detections"

    objs = [c3_objects_from_gt(x) for x in gt_s]
    c3 = train_c3(objs, gt_s, epochs=200, bs=16, lr=1e-3, patience=1000, aug=False, log=lambda *_: None)
    plans = plan_c3(c3, objs, gt_s)
    m3 = c3_metrics(gt_s, plans)
    mean_path = np.mean([x.ego_fut for x in gt_s], 0)
    base = np.mean([np.linalg.norm(mean_path[5] - x.ego_fut[5]) for x in gt_s])
    print(f"C3 overfit: L2@3s {m3['L2@3s']:.3f} m vs mean-path {base:.3f} m, collisions {m3['collision_rate']}")
    assert m3["L2@3s"] < 0.5 * base

    c3e = train_c3(objs, gt_s, epochs=150, bs=16, lr=1e-3, patience=1000, aug=True, use_ego=True,
                   log=lambda *_: None)
    m3e = c3_metrics(gt_s, plan_c3(c3e, objs, gt_s))
    print(f"C3 with ego path + augmentation: L2@3s {m3e['L2@3s']:.3f} m")
    assert m3e["L2@3s"] < 0.5 * base
    aug = D.augment(gt_s[3], np.random.default_rng(1))
    for _ in range(20):
        aug = D.augment(gt_s[3], np.random.default_rng(_))
        f, a = aug._aug
        assert np.allclose(np.linalg.norm(aug.fut - aug.cur_xy[:, None], axis=-1),
                           np.linalg.norm(gt_s[3].fut - gt_s[3].cur_xy[:, None], axis=-1), atol=1e-4)
        assert aug.command == (2 - gt_s[3].command if f else gt_s[3].command)
    print("augmentation keeps distances and swaps the command on mirror: OK")
    p1, p2 = plan_c3(c3, objs, gt_s), plan_c3(c3, objs, gt_s)
    assert np.array_equal(p1, p2), "evaluation must be deterministic"
    print("determinism: OK; object use:", round(object_use(plans, plan_c3(c3, objs, gt_s, drop_objects=True)), 3))
    pd = predict_c2(c2, det_s)
    print("C2 output change gt vs det:", c2_output_change(gt_s, pr, det_s, pd))
    _ = [c3_objects_from_c2(x, p) for x, p in zip(det_s, pd)]
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
