"""Generate all full-trainval result figures with matplotlib.

Uses the numeric data from Kernel A (timing/splits) and Kernel B (campaign +
baseline). Produces publication-style PNGs into ./images/.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)

BEFORE = "#2b6cb0"   # blue
AFTER = "#dd6b20"    # orange
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.grid": False})

baseline = json.load(open(os.path.join(HERE, "baseline_fulltrainval.json")))

# 15-update campaign (from Kernel B log)
CAMP = [
    ("full_ep10_s1", 0.1186, -2.172, 0.124, 1.0524),
    ("full_ep10_s2", 0.1217, -2.279, 0.193, 1.0177),
    ("full_ep10_s3", 0.1285, -2.149, 0.187, 0.9220),
    ("half_ep10_s1", 0.1135, -2.550, -0.121, 1.3518),
    ("half_ep10_s2", 0.1198, -2.706, -0.230, 1.1880),
    ("half_ep10_s3", 0.1064, -1.773, -0.151, 1.4556),
    ("quarter_ep10_s1", 0.1209, -2.711, 0.073, 0.7870),
    ("quarter_ep10_s2", 0.1134, -1.286, -0.093, 0.7789),
    ("quarter_ep10_s3", 0.1010, -2.816, 0.008, 1.0009),
    ("full_ep5_s1", 0.1381, -1.083, -0.095, 0.8894),
    ("full_ep5_s2", 0.1224, -1.882, -0.130, 1.2150),
    ("full_ep5_s3", 0.1301, -1.083, -0.041, 0.9168),
    ("full_ep20_s1", 0.1416, -2.594, 0.037, 1.3700),
    ("full_ep20_s2", 0.1258, -1.772, -0.083, 1.4978),
    ("full_ep20_s3", 0.1304, -2.445, -0.020, 1.2030),
]
tags = [c[0] for c in CAMP]
d1 = np.array([c[1] for c in CAMP])
d2 = np.array([c[2] for c in CAMP])
d3 = np.array([c[3] for c in CAMP])
shift = np.array([c[4] for c in CAMP])


def save(fig, name):
    p = os.path.join(IMG, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ── Fig 1: baseline cascade signature (isolated vs pipeline, C2 and C3) ──
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
ax = axes[0]
vals = [baseline["c2_iso_minADE"], baseline["c2_pipe_minADE"]]
ax.bar(["C2 isolated", "C2 pipeline"], vals, color=[BEFORE, AFTER])
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("minADE (m)")
ax.set_title("C2 prediction: cascade gap")
ax = axes[1]
vals = [baseline["c3_iso_L2"], baseline["c3_pipe_L2"]]
ax.bar(["C3 isolated", "C3 pipeline"], vals, color=[BEFORE, AFTER])
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("L2 @ 3s (m)")
ax.set_title("C3 planning: pipeline > isolated")
fig.suptitle("Baseline cascade signature — full trainval (n=2020 frames)", fontsize=12)
save(fig, "fig_baseline_cascade.png")


# ── Fig 2: campaign delta1 (all 15 improve) ──
fig, ax = plt.subplots(figsize=(9, 3.6))
colors = [AFTER if x > 0 else BEFORE for x in d1]
ax.bar(range(15), d1, color=AFTER)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(15))
ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=8)
ax.set_ylabel(r"$\delta_1$ (C1 mAP change)")
ax.set_title(r"C1 improves on its own metric in all 15 runs ($\delta_1 > 0$)")
save(fig, "fig_campaign_delta1.png")


# ── Fig 3: campaign delta3 (planner worse in 6) — strict-EE highlight ──
fig, ax = plt.subplots(figsize=(9, 3.6))
colors = ["#c53030" if x > 0 else "#2f855a" for x in d3]  # red=worse, green=better
ax.bar(range(15), d3, color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(15))
ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=8)
ax.set_ylabel(r"$\Delta_3$ (C3 pipe L2 change)")
ax.set_title(r"Planner degrades ($\Delta_3>0$, red) in 6/15 — strict entangled enhancement")
save(fig, "fig_campaign_delta3.png")


# ── Fig 4: strict-EE scatter (delta1 vs delta3) ──
fig, ax = plt.subplots(figsize=(6, 5))
strict = (d1 > 0) & (d3 > 0)
ax.scatter(d1[~strict], d3[~strict], c=BEFORE, s=60, label="benign / helped", zorder=3)
ax.scatter(d1[strict], d3[strict], c="#c53030", s=90, marker="*",
           label="strict entangled enhancement", zorder=4)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel(r"$\delta_1$  (C1 improves $\rightarrow$)")
ax.set_ylabel(r"$\Delta_3$  (planner worse $\uparrow$)")
ax.set_title("Strict entangled enhancement: C1 better, system worse\n(top-right quadrant)")
ax.legend(loc="lower left")
save(fig, "fig_strict_ee_scatter.png")


# ── Fig 5: plan-shift per update ──
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.bar(range(15), shift, color=BEFORE)
ax.set_xticks(range(15))
ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("plan shift @3s (m)")
ax.set_title(f"Plan shift per update (range {shift.min():.2f}–{shift.max():.2f} m)")
save(fig, "fig_plan_shift.png")


# ── Fig 6: per-scene C3 pipeline L2 distribution (the n=2020 richness) ──
iso = [s["L2@3s"] for s in baseline["c3_iso_per_scene"]]
pipe = [s["L2@3s"] for s in baseline["c3_pipe_per_scene"]]
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, max(max(iso), max(pipe)), 25)
ax.hist(iso, bins=bins, alpha=0.6, color=BEFORE, label=f"C3 isolated (mean {np.mean(iso):.2f})")
ax.hist(pipe, bins=bins, alpha=0.6, color=AFTER, label=f"C3 pipeline (mean {np.mean(pipe):.2f})")
ax.set_xlabel("L2 @ 3s (m)")
ax.set_ylabel("number of scenes")
ax.set_title("Per-scene planning error across 50 Singapore val scenes")
ax.legend()
save(fig, "fig_per_scene_l2.png")


# ── Fig 7: agents per scene vs error (context) ──
fig, ax = plt.subplots(figsize=(6.5, 4.5))
nag = [s["n_agents"] for s in baseline["c3_pipe_per_scene"]]
col = ["#c53030" if s["collided"] else BEFORE for s in baseline["c3_pipe_per_scene"]]
ax.scatter(nag, pipe, c=col, s=40, alpha=0.8)
ax.set_xlabel("agents in scene")
ax.set_ylabel("C3 pipeline L2 @3s (m)")
ax.set_title("Scene complexity vs planning error (red = collision)")
save(fig, "fig_agents_vs_error.png")

print("\nAll figures written to", IMG)
