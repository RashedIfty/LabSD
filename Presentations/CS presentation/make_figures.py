"""Figures for the CS presentation, read from the E1_new result JSONs.

Run with the project venv:
    venv/bin/python "Presentations/CS presentation/make_figures.py"
Writes fig_delta2.png, fig_delta3.png and fig_retrain.png next to this file and prints every number used on the result slides.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent / "experiments" / "E1_new" / "results"
C = {X: json.load(open(R / f"stage2/campaign_base_s{X}.json")) for X in (0, 1)}
B = {X: json.load(open(R / f"stage3/bootstrap_base_s{X}.json")) for X in (0, 1)}

# Seed variation thresholds, exactly as in experiments/E1_new/analysis/make_results_doc.py
TAU2 = max(abs(C[X]["seed_noise"]["settings"][s]["delta2_minADE"]) for X in (0, 1) for s in ("old_pipeline", "isolated"))
TAU3 = max(abs(C[X]["seed_noise"]["settings"][s]["delta3_L2@3s"]["ego"]) for X in (0, 1) for s in ("old_pipeline", "isolated"))

rows = []
for X in (0, 1):
    for name, u in C[X]["updates"].items():
        op, gt = u["settings"]["old_pipeline"], u["settings"]["isolated"]
        bs = B[X]["updates"][name]
        rows.append(dict(base=X, frac=u["frac"], epochs=u["epochs"], seed=u["seed"], d1=u["delta1_mAP50"],
                         d2c=op["delta2_minADE"], d2g=gt["delta2_minADE"],
                         d3c=op["delta3_L2@3s"]["ego"], d3g=gt["delta3_L2@3s"]["ego"],
                         s2c=bs["old_pipeline"]["delta2_minADE"]["harm"], s2g=bs["isolated"]["delta2_minADE"]["harm"],
                         s3c=bs["old_pipeline"]["delta3_L2@3s"]["harm"], s3g=bs["isolated"]["delta3_L2@3s"]["harm"]))


def count(f):
    return sum(1 for r in rows if f(r))


def numbers():
    """Every number shown on the result slides (c = C1-output training, g = GT training)."""
    n = {"tau2": TAU2, "tau3": TAU3,
         "d1_min": min(r["d1"] for r in rows), "d1_max": max(r["d1"] for r in rows),
         "d1_pos": count(lambda r: r["d1"] > 0)}
    for m in ("c", "g"):
        n[f"d2{m}_any"] = count(lambda r: r[f"d2{m}"] > 0)
        n[f"d2{m}_tau"] = count(lambda r: r[f"d2{m}"] > TAU2)
        n[f"d2{m}_sig"] = count(lambda r: r[f"s2{m}"])
        n[f"d3{m}_any"] = count(lambda r: r[f"d3{m}"] > 0)
        n[f"d3{m}_tau"] = count(lambda r: r[f"d3{m}"] > TAU3)
        n[f"d3{m}_sig"] = count(lambda r: r[f"s3{m}"])
    n["c_worse_than_g"] = count(lambda r: r["d2c"] > r["d2g"])
    n["tau_from_seed0"] = count(lambda r: r["d2c"] > TAU2 and r["base"] == 0)
    for X in (0, 1):
        sub = [r["d2c"] for r in rows if r["base"] == X]
        n[f"mean_d2c_s{X}"] = sum(sub) / len(sub)
    n["d3_max_abs"] = max(max(abs(r["d3c"]), abs(r["d3g"])) for r in rows)
    for X in (0, 1):
        for m in ("c", "g"):
            sub = [r[f"d3{m}"] for r in rows if r["base"] == X]
            n[f"mean_d3{m}_s{X}"] = sum(sub) / len(sub)
    return n


# ───────────────────────── style ─────────────────────────
DF = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf", "Cambriai.ttf"):
    try:
        font_manager.fontManager.addfont(DF + f)
    except Exception:
        pass
plt.rcParams.update({"font.family": ["Cambria", "Times New Roman", "serif"], "font.size": 13,
                     "axes.edgecolor": "#6B7280", "axes.linewidth": 0.8})
NAVY, GOLD, WINE, GREY = "#0B2545", "#B08D57", "#8B2E2E", "#6B7280"


def fig_delta(key, tau, ylabel, out_name, pad):
    """Delta per retraining for both training modes, with the seed-variation threshold."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=200)
    x = list(range(1, 31))
    c, g, sg = f"{key}c", f"{key}g", f"s{key[-1]}c"
    hi = max(max(r[c], r[g]) for r in rows)
    lo = min(min(r[c], r[g]) for r in rows)
    ax.axhspan(tau, hi + pad * 2, color=WINE, alpha=0.07, lw=0)
    ax.axhline(tau, color=WINE, lw=1.1, ls="--", label=f"seed variation ({tau:.3f} m)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(15.5, color=GREY, lw=0.8, ls=":")
    ax.scatter(x, [r[g] for r in rows], s=42, marker="D", facecolor="white", edgecolor=GOLD, lw=1.6,
               label="GT training", zorder=3)
    ax.scatter(x, [r[c] for r in rows], s=48, color=NAVY, label="C$_1$-output training", zorder=4)
    sig = [(i + 1, r[c]) for i, r in enumerate(rows) if r[sg]]
    sig += [(i + 1, r[g]) for i, r in enumerate(rows) if r[f"s{key[-1]}g"]]
    ax.scatter([a for a, _ in sig], [b for _, b in sig], s=190, facecolor="none", edgecolor=WINE, lw=1.6,
               label="statistically significant", zorder=5)
    ax.set_ylim(lo - pad * 1.2, hi + pad)
    ax.set_xlim(0.3, 30.7)
    ax.text(8, lo - pad * 0.95, "original C$_1$ with seed 0", ha="center", color=GREY, fontsize=11.5, style="italic")
    ax.text(23, lo - pad * 0.95, "original C$_1$ with seed 1", ha="center", color=GREY, fontsize=11.5, style="italic")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Retraining of C$_1$")
    ax.set_ylabel(ylabel)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10.5, ncol=4,
              handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    out = HERE / out_name
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def retrain_numbers():
    """Stage 1b: one retraining of each original C1, C2/C3 kept fixed or retrained on the output of C1'."""
    out = {}
    for X in (0, 1):
        d = json.load(open(R / f"stage1b/p1_c1s{X}_seed0.json"))["settings"]
        op, npl = d["old_pipeline"], d["new_pipeline"]
        out[X] = {"c2": [op["c2_pipe"][f"c1_old_s{X}"]["minADE"], op["c2_pipe"][f"c1_new_s{X}"]["minADE"],
                         npl["c2_pipe"][f"c1_new_s{X}"]["minADE"]],
                  "c3": [op["c3"]["ego"]["pipe"][f"c1_old_s{X}"]["c3"]["L2@3s"],
                         op["c3"]["ego"]["pipe"][f"c1_new_s{X}"]["c3"]["L2@3s"],
                         npl["c3"]["ego"]["pipe"][f"c1_new_s{X}"]["c3"]["L2@3s"]]}
    return out


def fig_retrain():
    """Grouped bars: before the retraining, after it with C2/C3 fixed, after retraining C2/C3 on C1' output."""
    nums = retrain_numbers()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), dpi=200)
    labels = ["Before: C$_1$, C$_1$-output training",
              "After: C$_1'$, C$_2$ and C$_3$ kept fixed",
              "After: C$_1'$, C$_2$ and C$_3$ retrained on the output of C$_1'$"]
    colors = ["#D9D4C7", WINE, NAVY]
    hi_pad = {"c2": 0.04, "c3": 0.06}
    for ax, key, title, ylab in ((axes[0], "c2", "Prediction model C$_2$", "minADE (m)"),
                                 (axes[1], "c3", "Planning model C$_3$", "error at 3 s (m)")):
        for X in (0, 1):
            for k in range(3):
                v = nums[X][key][k]
                ax.bar(X + (k - 1) * 0.26, v, 0.24, color=colors[k], label=labels[k] if X == 0 else None,
                       edgecolor="white")
                ax.text(X + (k - 1) * 0.26, v + hi_pad[key], f"{v:.2f}", ha="center", va="bottom", fontsize=10.5)
        hi = max(max(nums[X][key]) for X in (0, 1))
        ax.set_ylim(0, hi * 1.12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["original C$_1$\nwith seed 0", "original C$_1$\nwith seed 1"], fontsize=11.5)
        ax.set_title(title, fontsize=13.5, color=NAVY, fontweight="bold")
        ax.set_ylabel(ylab)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=1, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    out = HERE / "fig_retrain.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


MEASURES = [  # (key in the result JSON, plain name, model whose output is compared)
    ("objects_per_image_JS", "Number of objects per image", "C1"),
    ("type_mix_JS", "Mix of object classes", "C1"),
    ("confidence_W", "Confidence values (average shift)", "C1"),
    ("confidence_KS", "Confidence values (largest gap)", "C1"),
    ("forward_position_W_m", "Forward positions of objects", "C1"),
    ("lateral_position_W_m", "Lateral positions of objects", "C1"),
    ("only_one_version_fraction", "Objects found by only one version", "C1"),
    ("movement_W", "Length of predicted movements", "C2"),
    ("same_object_shift", "Shift of predicted positions", "C2"),
]


def detection_numbers():
    """Spearman rank correlation of each label-free measure with delta2 (C1-output training):
    over all 30 retrainings, and within the 15 retrainings of each original C1."""
    from scipy.stats import spearmanr
    vals = {k: [] for k, _, _ in MEASURES}
    vals["delta1"], d2, base = [], [], []
    for X in (0, 1):
        for u in C[X]["updates"].values():
            op = u["settings"]["old_pipeline"]
            for k, _, m in MEASURES:
                vals[k].append(u["h2_c1_update"][k] if m == "C1" else op["h2_c2_update"][k])
            vals["delta1"].append(u["delta1_mAP50"])
            d2.append(op["delta2_minADE"])
            base.append(X)
    out = {}
    for k, v in vals.items():
        pooled = spearmanr(v, d2)[0]
        per = [spearmanr([a for a, b in zip(v, base) if b == X], [a for a, b in zip(d2, base) if b == X])[0]
               for X in (0, 1)]
        out[k] = (pooled, per[0], per[1])
    return out


def fig_detection():
    nums = detection_numbers()
    rows_ = [(name, nums[k]) for k, name, _ in MEASURES] + [("δ$_1$ (needs ground truth)", nums["delta1"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=200)
    y = list(range(len(rows_)))[::-1]
    ax.axvline(0, color=GREY, lw=0.8)
    for yy, (name, (pl, s0, s1)) in zip(y, rows_):
        ax.plot([min(pl, s0, s1), max(pl, s0, s1)], [yy, yy], color="#D9D4C7", lw=2, zorder=1)
    ax.scatter([r[1][1] for r in rows_], y, marker="D", s=40, facecolor="white", edgecolor=GOLD, lw=1.5,
               label="within original C$_1$ with seed 0", zorder=3)
    ax.scatter([r[1][2] for r in rows_], y, marker="s", s=38, color=GREY, label="within original C$_1$ with seed 1",
               zorder=3)
    ax.scatter([r[1][0] for r in rows_], y, s=60, color=NAVY, label="over all 30 retrainings", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows_], fontsize=11.5)
    ax.axhline(2.5, color=GREY, lw=0.6, ls=":")
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.text(1.02, 6.0, "output\nof C$_1$", transform=ax.get_yaxis_transform(), fontsize=10.5, color=GREY,
            va="center", style="italic")
    ax.text(1.02, 1.5, "output\nof C$_2$", transform=ax.get_yaxis_transform(), fontsize=10.5, color=GREY,
            va="center", style="italic")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Rank correlation with Δ$_2$")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=10.5, ncol=3, handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "fig_detection.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    for k, v in numbers().items():
        print(f"{k:16s} {v:+.3f}" if isinstance(v, float) else f"{k:16s} {v}")
    fig_delta("d2", TAU2, "Δ$_2$: increase of the prediction error (m)", "fig_delta2.png", 0.05)
    for X, v in retrain_numbers().items():
        print(f"retrain s{X}  C2 {v['c2'][0]:.3f} {v['c2'][1]:.3f} {v['c2'][2]:.3f}   C3 {v['c3'][0]:.3f} {v['c3'][1]:.3f} {v['c3'][2]:.3f}")
    fig_retrain()
    for k, (pl, s0, s1) in detection_numbers().items():
        print(f"rho {k:28s} pooled {pl:+.2f}  seed0 {s0:+.2f}  seed1 {s1:+.2f}")
    fig_detection()
    fig_delta("d3", TAU3, "Δ$_3$: increase of the planning error (m)", "fig_delta3.png", 0.025)
