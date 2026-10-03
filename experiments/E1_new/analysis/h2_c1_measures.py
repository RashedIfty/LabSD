"""H2 label-free measures on the C1 output (Meeting 6 report, Table 4).

Compares two versions of C1 on the same singapore_test images using only their
detections (no ground truth). Run on the Stage 0 detection dumps:
    python h2_c1_measures.py <detections_dir>
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp

CLASSES = ["car", "truck", "bus", "trailer", "construction_vehicle",
           "motorcycle", "bicycle", "pedestrian"]
MATCH_RADIUS_M = 2.0   # same class and within 2 m in the ego frame counts as the same object


def js(p, q):
    """Jensen-Shannon divergence (base 2, 0 = equal, 1 = disjoint)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)


def per_image(recs):
    by = defaultdict(list)
    for r in recs:
        by[r["sample_token"]].append(r)
    return by


def only_one_version_fraction(a_by, b_by, images):
    """Fraction of objects found by only one of the two versions (greedy matching)."""
    unmatched, total = 0, 0
    for img in images:
        a, b = list(a_by.get(img, [])), list(b_by.get(img, []))
        total += len(a) + len(b)
        used = set()
        for da in a:
            best, best_d = None, MATCH_RADIUS_M
            for j, db in enumerate(b):
                if j in used or db["cls"] != da["cls"]:
                    continue
                d = ((da["x"] - db["x"]) ** 2 + (da["y"] - db["y"]) ** 2) ** 0.5
                if d <= best_d:
                    best, best_d = j, d
            if best is None:
                unmatched += 1
            else:
                used.add(best)
        unmatched += len(b) - len(used)
    return unmatched / total if total else 0.0


def compare(a, b, images):
    a_by, b_by = per_image(a), per_image(b)
    ca = [len(a_by.get(i, [])) for i in images]
    cb = [len(b_by.get(i, [])) for i in images]
    m = max(ca + cb) + 1
    ka, kb = Counter(r["cls"] for r in a), Counter(r["cls"] for r in b)
    sa, sb = [r["score"] for r in a], [r["score"] for r in b]
    return {
        "objects_per_image_JS": js(np.bincount(ca, minlength=m) + 1e-9,
                                   np.bincount(cb, minlength=m) + 1e-9),
        "mean_objects_per_image": [float(np.mean(ca)), float(np.mean(cb))],
        "type_mix_JS": js([ka[c] + 1e-9 for c in CLASSES], [kb[c] + 1e-9 for c in CLASSES]),
        "confidence_W": float(wasserstein_distance(sa, sb)),
        "confidence_KS": float(ks_2samp(sa, sb).statistic),
        "forward_position_W_m": float(wasserstein_distance([r["x"] for r in a], [r["x"] for r in b])),
        "lateral_position_W_m": float(wasserstein_distance([r["y"] for r in a], [r["y"] for r in b])),
        "only_one_version_fraction": only_one_version_fraction(a_by, b_by, images),
    }


def main(det_dir):
    det = {p.name.replace("_singapore_test.json", ""): json.load(open(p))
           for p in sorted(Path(det_dir).glob("*_singapore_test.json"))}
    # Images with no detection in any version are absent from every dump; use the union.
    images = sorted({r["sample_token"] for recs in det.values() for r in recs})
    pairs = {
        "seed_noise_old (old_s0 vs old_s1)": ("c1_old_s0", "c1_old_s1"),
        "seed_noise_new (new_s0 vs new_s1)": ("c1_new_s0", "c1_new_s1"),
        "update_s0 (old_s0 vs new_s0)": ("c1_old_s0", "c1_new_s0"),
        "update_s1 (old_s1 vs new_s1)": ("c1_old_s1", "c1_new_s1"),
    }
    out = {"n_images_with_any_detection": len(images),
           "match_radius_m": MATCH_RADIUS_M,
           "pairs": {k: compare(det[a], det[b], images) for k, (a, b) in pairs.items()}}
    return out


if __name__ == "__main__":
    res = main(sys.argv[1])
    print(json.dumps(res, indent=1))
