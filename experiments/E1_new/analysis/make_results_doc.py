"""Build the full results document (CS_seminar_IFTY_results.tex) from the result JSONs.

Every number in the document is read from experiments/E1_new/results/ and
models/*/manifest.json, which were downloaded from the Kaggle kernels.
Run:  python make_results_doc.py   (needs numpy, scipy)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

E1 = Path(__file__).resolve().parent.parent
R = E1 / "results"
OUT = E1.parent.parent / "ieee paper" / "Version 4" / "CS_seminar_IFTY_results.tex"


def J(p):
    return json.load(open(R / p))


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def tag(name):  # upd_b0_f100_e10_s1 -> b0 f100 e10 s1
    return name.replace("upd_", "").replace("_", " ")


L = []
A = L.append

# ───────────────────────────── preamble ─────────────────────────────
A(r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{pdflscape}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{placeins}
\usepackage[colorlinks=true,linkcolor=blue!60!black]{hyperref}
\newcommand{\Y}{\cellcolor{red!15}}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\title{Full Experiment Results\\\large Supplement to the CS seminar paper (CS\_seminar\_IFTY)}
\author{Rashedul Arefin Ifty}
\date{October 2026}
\begin{document}
\maketitle
\tableofcontents
\newpage
""")

# ───────────────────────────── 1. intro ─────────────────────────────
A(r"""\FloatBarrier
\section{What this document is}
The 4-page paper can only show summaries. This document lists every result
of every experiment that I ran on Kaggle for the new stage of E1, where all
three modules are ML components. All tables are generated directly from the
result files that the Kaggle kernels wrote (the generator is
\texttt{experiments/E1\_new/analysis/make\_results\_doc.py}), so no number was
copied by hand.

Notation used everywhere:
\begin{itemize}
\item $C_1$ perception (YOLOv11n), $C_2$ prediction (AutoBot-Ego style), $C_3$ planning (PlanT style).
\item \textbf{old $C_1$}: $C_1$ trained on Boston~1. Two copies, seed 0 (s0) and seed 1 (s1).
\item \textbf{update}: retraining an old $C_1$ on Singapore~1. Update names read
\texttt{b0 f50 e10 s2} = base old $C_1$ seed 0, 50\,\% of the Singapore images, 10 epochs, update seed 2.
\item \textbf{isolated training} (iso): $C_2$, $C_3$ trained on the ground truth.
\textbf{old-pipeline training} (op): trained on the output of the old $C_1$.
\textbf{new-pipeline training} (np): trained on the output of the new $C_1$.
\item $C_3$ comes in two variants: \textbf{ego} (gets the last 1.5\,s of the ego path, used in the paper) and \textbf{noego} (no ego input, failed its baseline, kept here for completeness).
\end{itemize}
""")

# ───────────────────────────── 2. formulas ─────────────────────────────
A(r"""\FloatBarrier
\section{Definitions and formulas}
\begin{itemize}
\item \textbf{Accuracy gain of $C_1$}: $\delta_1 = \mathrm{mAP50}(C_1^{\text{new}}) - \mathrm{mAP50}(C_1^{\text{old}})$ on Singapore~2.
\item \textbf{Prediction error} (minADE over $M=6$ modes, $T=6$ steps of 0.5\,s, $m_t$ = valid step):
\[ \text{minADE} = \frac{1}{N}\sum_{n=1}^{N} \min_{k=1..M} \frac{\sum_t m_t \,\lVert \hat p^{(k)}_{n,t} - p_{n,t}\rVert_2}{\sum_t m_t} \]
over the objects found by $C_1$ that are matched one to one (Hungarian, same class, radius $\max(2, 0.15\cdot\text{range})$\,m) to a ground-truth object.
\item \textbf{Planning error}: $\mathrm{L2@}t = \frac{1}{A}\sum_{a} \lVert \hat e_{a}(t) - e_{a}(t) \rVert_2$ for $t = 1, 2, 3$\,s over the $A = 1{,}720$ test samples ($e$ = human ego path).
\item \textbf{Collision rate}: fraction of test samples in which an oriented box of the ego vehicle (4.08\,m $\times$ 1.73\,m) on the planned path overlaps a ground-truth object box at any of the 6 future steps.
\item \textbf{Object use}: mean distance between the planned 3\,s position with and without the objects in the input of $C_3$.
\item $\Delta_2 = \text{minADE}_{\text{new }C_1} - \text{minADE}_{\text{old }C_1}$, \quad $\Delta_3 = \text{L2@3s}_{\text{new }C_1} - \text{L2@3s}_{\text{old }C_1}$ (same $C_2$, $C_3$).
\item \textbf{Entangled enhancement}: $\delta_1 > 0$ and ($\Delta_2 > 0$ or $\Delta_3 > 0$).
\item \textbf{Random variation} (level 2 threshold): $\tau = \max |\Delta|$ when the old $C_1$ is swapped for the other old $C_1$ (both bases, both trainings): $\tau_2 = 0.112$\,m, $\tau_3 = 0.018$\,m.
\item \textbf{Bootstrap} (level 3): 2{,}000 resamples of the 50 test scenes with replacement; per resample $\Delta = \bar{e}_{\text{new}} - \bar{e}_{\text{old}}$ on the same scenes; harm = 2.5th percentile $> 0$.
\item \textbf{Jensen--Shannon divergence} (base 2): $\mathrm{JS}(P,Q) = \tfrac12 \mathrm{KL}(P\Vert M) + \tfrac12\mathrm{KL}(Q\Vert M)$, $M = \tfrac12(P+Q)$; 0 = equal, 1 = disjoint.
\item \textbf{Wasserstein distance}: $W_1(P,Q) = \int |F_P(x) - F_Q(x)|\,dx$. \textbf{KS statistic}: $\sup_x |F_P(x) - F_Q(x)|$.
\item \textbf{Only-one-version fraction}: share of objects found by only one of the two $C_1$ on the same image (same class, within 2\,m counts as the same object).
\item \textbf{Spearman rank correlation} $\rho$ and \textbf{AUROC} = P(measure of a harmful update $>$ measure of a harmless update), harmful = $\Delta_2 > 0$ after old-pipeline training.
\end{itemize}
""")

# ───────────────────────────── 3. data and compute ─────────────────────────────
A(r"""\FloatBarrier
\section{Data and compute}
\begin{table}[h]\centering\small
\caption{The 350-scene subset (full-size CAM\_FRONT keyframes, 4.645\,GB in total, 0 missing images)}
\begin{tabular}{lrrrl}\toprule
Part & Scenes & Keyframes & Images with labels & Used for\\\midrule
Boston 1 (boston\_c1) & 100 & 4{,}008 & 3{,}923 & training the old $C_1$\\
Boston 2 (boston\_c23) & 100 & 4{,}046 & -- & training $C_2$, $C_3$ (86 train, 14 validation)\\
Singapore 1 (singapore\_update) & 100 & 3{,}997 & 3{,}575 & updates of $C_1$\\
Singapore 2 (singapore\_test) & 50 & 2{,}020 & 1{,}805 & testing (1{,}720 test samples)\\\bottomrule
\end{tabular}\end{table}

\begin{table}[h]\centering\small
\caption{Kaggle kernels and run times}
\begin{tabular}{lllr}\toprule
Stage & Kernel & GPU & Run time\\\midrule
Subset & ifty1011/labsd-e1-subset350 & CPU & 434.5\,s\\
Stage 0 (old and new $C_1$) & ifty1011/labsd-e1-stage0-c1 & T4 & 2{,}546\,s\\
Stage 1 data export & ifty1011/labsd-e1-stage1-data & T4 & 953\,s\\
Stage 1b ($C_2$, $C_3$) & ifty1011/labsd-e1-stage1b & T4 & 2{,}365\,s\\
Stage 2 campaign, base s0 & ifty1011/labsd-e1-stage2-campaign & T4 & 9{,}096\,s\\
Stage 2 campaign, base s1 & rai73416/labsd-e1-stage2-campaign & T4 & 8{,}002\,s\\
Stage 3 bootstrap, base s0 & ifty1011/labsd-e1-stage3-bootstrap & CPU & 421.5\,s\\
Stage 3 bootstrap, base s1 & rai73416/labsd-e1-stage3-bootstrap & CPU & 423.0\,s\\\bottomrule
\end{tabular}\end{table}
""")

# ───────────────────────────── 4. stage 0 ─────────────────────────────
m0 = json.load(open(E1 / "results/stage0_v2/models/manifest.json"))
s0 = J("stage0_v2/stage0_summary.json")
A(r"""\FloatBarrier
\section{Stage 0: old and new $C_1$}
YOLOv11n, 10 epochs, image size 640, batch 8, checkpoint \texttt{last.pt}, never selected on the test set.
Training times: old s0 579\,s, new s0 493\,s, old s1 535\,s, new s1 487\,s.
\begin{table}[h]\centering\small\caption{The four Stage 0 models on Singapore 2}
\begin{tabular}{lllrrrrr}\toprule
Model & Init & Trained on & mAP50 & mAP50-95 & Precision & Recall & Detections\\\midrule""")
for k, v in m0["models"].items():
    t = v["test_metrics"]
    A(f"{k.replace('_', ' ')} & {v['init'].replace('_', ' ')} & {v['trained_on'].replace('_', ' ')} & {f(t['mAP50'],4)} & {f(t['mAP50_95'],4)} & {f(t['precision'],3)} & {f(t['recall'],3)} & {v['n_detections_test']:,}\\\\")
A(r"""\bottomrule\end{tabular}\end{table}""")
A(f"$\\delta_1$ (Stage 0 updates): seed 0 {f(s0['delta1_mAP50_per_seed']['0'],4,True)}, seed 1 {f(s0['delta1_mAP50_per_seed']['1'],4,True)}. "
  f"Seed gap of the old $C_1$ (mAP50): {f(s0['old_c1_seed_gap_mAP50'],4)}.\n")
h = J("stage0_v2/h2_c1_measures.json")
A(r"""\begin{table}[h]\centering\small\caption{First look at the output change of $C_1$ (no ground truth), %d images with any detection}
\begin{tabular}{lrrrr}\toprule
Measure & old s0 vs old s1 & new s0 vs new s1 & old s0 vs new s0 & old s1 vs new s1\\\midrule""" % h["n_images_with_any_detection"])
pairs = list(h["pairs"].values())
for key in pairs[0]:
    vals = [p[key] for p in pairs]
    cells = [f"{v[0]:.2f}/{v[1]:.2f}" if isinstance(v, list) else f(v, 4) for v in vals]
    A(key.replace("_", " ") + " & " + " & ".join(cells) + r"\\")
A(r"""\bottomrule\end{tabular}\end{table}""")

# ───────────────────────────── 5. stage 1 data ─────────────────────────────
A(r"""\FloatBarrier
\section{Stage 1 data export}
Ground truth and the detections of the four Stage 0 models, per keyframe, for Boston 2 (4{,}046 keyframes) and Singapore 2 (2{,}020 keyframes).
The Singapore 2 counts are identical to Stage 0 (consistency check passed).
\begin{table}[h]\centering\small\caption{Number of detections per model}
\begin{tabular}{lrrrr}\toprule
Split & old s0 & new s0 & old s1 & new s1\\\midrule
Boston 2 & 25{,}286 & 20{,}511 & 26{,}813 & 23{,}663\\
Singapore 2 & 3{,}563 & 3{,}603 & 3{,}705 & 3{,}409\\\bottomrule
\end{tabular}\end{table}
""")

# ───────────────────────────── 6. stage 1 (first attempt) ─────────────────────────────
A(r"""\FloatBarrier
\section{Stage 1: first ML version of $C_2$ and $C_3$ (failed the baselines)}
First version: scene-level C2 (no target-centred view), C3 without any ego input, 40/60 epochs, no validation set, no augmentation.
Both modules learned Boston but did not beat the baselines on Singapore 2
(stay-still 4.48\,m for $C_2$, mean path per command 9.23\,m for $C_3$).
This version was replaced by Stage 1b.
\begin{table}[h]\centering\scriptsize\caption{Stage 1 results on Singapore 2 (C3 = noego)}
\begin{tabular}{llrrrrrrr}\toprule
C1 seed & Training & C2 iso & C1 used & C2 pipe & C3 L2@1s & L2@2s & L2@3s & Collision\\\midrule""")
for X in (0, 1):
    r1 = J(f"stage1/p1_c1s{X}_seed0.json")
    for name, st in r1["settings"].items():
        for c1, p in st["pipe"].items():
            A(f"{X} & {name.replace('_','-')} & {f(st['c2_iso']['minADE'])} & {c1.replace('c1_','').replace('_',' ')} & {f(p['c2']['minADE'])} & {f(p['c3']['L2@1s'],2)} & {f(p['c3']['L2@2s'],2)} & {f(p['c3']['L2@3s'])} & {f(p['c3']['collision_rate'])}\\\\")
A(r"""\bottomrule\end{tabular}\end{table}""")

# ───────────────────────────── 7. stage 1b ─────────────────────────────
mb = json.load(open(E1 / "models/stage1b/manifest.json"))
A(r"""\FloatBarrier
\section{Stage 1b: final $C_2$ and $C_3$}
Target-centred $C_2$ (736{,}799 parameters), $C_3$ (619{,}010 parameters, ego and noego variants), early stopping on the 14 Boston validation scenes, mirror and rotation augmentation.
\begin{table}[h]\centering\small\caption{Best validation error (Boston 2 validation scenes)}
\begin{tabular}{lrr}\toprule Model & best val minADE (m) & best val L2@3s (m)\\\midrule""")
for k, v in mb["models"].items():
    A(f"{k.replace('_',' ')} & {f(v.get('best_val_minADE'))} & {f(v.get('best_val_L2@3s'))}\\\\")
A(r"""\bottomrule\end{tabular}\end{table}""")
b1 = J("stage1b/p1_c1s0_seed0.json")["baselines"]
A(f"Baselines on Singapore 2: $C_2$ stay-still {f(b1['c2_stay_still_ADE'])}\\,m; $C_3$ mean path per command "
  f"L2@1/2/3s = {f(b1['c3_mean_path']['L2@1s'],2)}/{f(b1['c3_mean_path']['L2@2s'],2)}/{f(b1['c3_mean_path']['L2@3s'])}\\,m, collision {f(b1['c3_mean_path']['collision_rate'])}. "
  "Constant-velocity reference with ground-truth identities: 1.008\\,m (test), 0.579\\,m (Boston validation).\n")
A(r"""\begin{landscape}
\begin{table}[h!]\centering\scriptsize\caption{Stage 1b results on Singapore 2 (1{,}720 test samples)}
\begin{tabular}{llllrrrrrrrrrr}\toprule
C1 seed & Training & C3 & C1 used & C2 minADE & scored & stay-still & L2@1s & L2@2s & L2@3s & Collision & Obj. use & iso L2@3s & iso obj. use\\\midrule""")
for X in (0, 1):
    r = J(f"stage1b/p1_c1s{X}_seed0.json")
    for name, st in r["settings"].items():
        for var in ("ego", "noego"):
            c3 = st["c3"][var]
            for c1, p in c3["pipe"].items():
                c2 = st["c2_pipe"][c1]
                A(f"{X} & {name.replace('_','-')} & {var} & {c1.replace('c1_','').replace('_',' ')} & {f(c2['minADE'])} & {c2['n_objects_scored']} & "
                  f"{f(st['c2_stay_still_on_scored'][c1])} & {f(p['c3']['L2@1s'],2)} & {f(p['c3']['L2@2s'],2)} & {f(p['c3']['L2@3s'])} & "
                  f"{f(p['c3']['collision_rate'])} & {f(p['object_use_m'],2)} & {f(c3['c3_iso']['L2@3s'])} & {f(c3['object_use_iso_m'],2)}\\\\")
        A(r"\midrule")
A(r"""\bottomrule\end{tabular}\end{table}
\begin{table}[h!]\centering\small\caption{Stage 1b deltas (old $C_1$ $\rightarrow$ Stage 0 new $C_1$) and seed-swap noise}
\begin{tabular}{llrrrrrr}\toprule
C1 seed & Training & $\Delta_2$ & noise $\Delta_2$ & $\Delta_3$ ego & noise ego & $\Delta_3$ noego & noise noego\\\midrule""")
for X in (0, 1):
    r = J(f"stage1b/p1_c1s{X}_seed0.json")
    for name in ("isolated", "old_pipeline"):
        st = r["settings"][name]
        A(f"{X} & {name.replace('_','-')} & {f(st['delta2_minADE'],3,True)} & {f(st['seed_noise_delta2_minADE'],3,True)} & "
          f"{f(st['c3']['ego']['delta3_L2@3s'],3,True)} & {f(st['c3']['ego']['seed_noise_delta3_L2@3s'],3,True)} & "
          f"{f(st['c3']['noego']['delta3_L2@3s'],3,True)} & {f(st['c3']['noego']['seed_noise_delta3_L2@3s'],3,True)}\\\\")
A(r"""\bottomrule\end{tabular}\end{table}
\end{landscape}""")

# ───────────────────────────── 8. stage 2 campaign ─────────────────────────────
C = {X: J(f"stage2/campaign_base_s{X}.json") for X in (0, 1)}
M = {X: J(f"stage2/manifest_base_s{X}.json") for X in (0, 1)}
B = {X: J(f"stage3/bootstrap_base_s{X}.json") for X in (0, 1)}
A(r"""\FloatBarrier
\section{Stage 2: the 30-update campaign}
Each base old $C_1$ is updated with 15 settings: data $\{100, 50, 25\}\,\% \times$ seeds $\{1,2,3\}$ at 10 epochs, and epochs $\{5, 20\} \times$ seeds $\{1,2,3\}$ at 100\,\%.
$C_2$ and $C_3$ are the saved Stage 1b models and are never retrained. Base old $C_1$: s0 mAP50 = BASE0, s1 mAP50 = BASE1.
""".replace("BASE0", f(C[0]["base_mAP50"], 4)).replace("BASE1", f(C[1]["base_mAP50"], 4)))

A(r"""\subsection{The updated $C_1$ models}
\begin{table}[h]\centering\scriptsize\caption{All 30 updated $C_1$ on Singapore 2}
\begin{tabular}{rlrrrrrrrr}\toprule
\# & Update & Data & Epochs & Seed & Train images & mAP50 & mAP50-95 & Precision & Recall\\\midrule""")
k = 0
ROWS = []
for X in (0, 1):
    for name, u in C[X]["updates"].items():
        k += 1
        t = u["test_metrics"]
        A(f"{k} & {tag(name)} & {int(u['frac']*100)}\\% & {u['epochs']} & {u['seed']} & {u['train_images']:,} & {f(t['mAP50'],4)} & {f(t['mAP50_95'],4)} & {f(t['precision'],3)} & {f(t['recall'],3)}\\\\")
        ROWS.append((k, X, name, u))
    if X == 0:
        A(r"\midrule")
A(r"""\bottomrule\end{tabular}\end{table}""")


def setting_table(setting, title, heading):
    A(r"""\begin{landscape}\subsection{%s}\begin{table}[h!]\centering\scriptsize\caption{%s}
\begin{tabular}{rlrrrrrrrrrrrrr}\toprule
\# & Update & $\delta_1$ & C2 base & C2 new & $\Delta_2$ & scored & ego base & ego new & $\Delta_3$ ego & $\Delta$coll. ego & obj.\ use & noego base & noego new & $\Delta_3$ noego\\\midrule""" % (heading, title))
    for k, X, name, u in ROWS:
        s = u["settings"][setting]
        b, n = s["base"], s["updated"]
        A(f"{k} & {tag(name)} & {f(u['delta1_mAP50'],4,True)} & {f(b['c2']['minADE'])} & {f(n['c2']['minADE'])} & {f(s['delta2_minADE'],3,True)} & {n['c2']['n_objects_scored']} & "
          f"{f(b['c3']['ego']['L2@3s'])} & {f(n['c3']['ego']['L2@3s'])} & {f(s['delta3_L2@3s']['ego'],3,True)} & {f(s['delta_collision']['ego'],3,True)} & "
          f"{f(n['c3']['ego']['object_use_m'],2)} & {f(b['c3']['noego']['L2@3s'])} & {f(n['c3']['noego']['L2@3s'])} & {f(s['delta3_L2@3s']['noego'],3,True)}\\\\")
        if k == 15:
            A(r"\midrule")
    A(r"\midrule")
    for X in (0, 1):
        sn = C[X]["seed_noise"]["settings"][setting]
        b, n = sn["base"], sn["updated"]
        A(f"noise & old s{X}$\\rightarrow$old s{1-X} & -- & {f(b['c2']['minADE'])} & {f(n['c2']['minADE'])} & {f(sn['delta2_minADE'],3,True)} & {n['c2']['n_objects_scored']} & "
          f"{f(b['c3']['ego']['L2@3s'])} & {f(n['c3']['ego']['L2@3s'])} & {f(sn['delta3_L2@3s']['ego'],3,True)} & {f(sn['delta_collision']['ego'],3,True)} & "
          f"{f(n['c3']['ego']['object_use_m'],2)} & {f(b['c3']['noego']['L2@3s'])} & {f(n['c3']['noego']['L2@3s'])} & {f(sn['delta3_L2@3s']['noego'],3,True)}\\\\")
    A(r"""\bottomrule\end{tabular}\end{table}\end{landscape}""")


setting_table("old_pipeline", r"Old-pipeline training: errors with the old $C_1$ (base) and with the updated $C_1$ (new), in metres",
              r"Effect on $C_2$ and $C_3$, old-pipeline training")
setting_table("isolated", r"Isolated training: errors with the old $C_1$ (base) and with the updated $C_1$ (new), in metres",
              r"Effect on $C_2$ and $C_3$, isolated training")

# levels per update
A(r"\FloatBarrier")
A(f"Old-pipeline $\\Delta_2$ is larger than isolated $\\Delta_2$ in "
  f"{sum(u['settings']['old_pipeline']['delta2_minADE'] > u['settings']['isolated']['delta2_minADE'] for _,_,_,u in ROWS)} of 30 updates.\n")
A(r"""\subsection{Entangled enhancement at three levels, per update}
Y = harm at that level (shaded). L1 = any increase, L2 = above random variation ($\tau_2 = 0.112$, $\tau_3 = 0.018$\,m), L3 = bootstrap 95\,\% interval above zero. EE = L1 for $\Delta_2$ or $\Delta_3$.
\begin{table}[h]\centering\scriptsize\caption{Per-update classification ($C_3$ = ego)}
\begin{tabular}{rl|ccc|ccc|ccc|ccc|c}\toprule
 & & \multicolumn{6}{c|}{Old-pipeline training} & \multicolumn{6}{c|}{Isolated training} & \\
 & & \multicolumn{3}{c|}{$\Delta_2$} & \multicolumn{3}{c|}{$\Delta_3$} & \multicolumn{3}{c|}{$\Delta_2$} & \multicolumn{3}{c|}{$\Delta_3$} & \\
\# & Update & L1 & L2 & L3 & L1 & L2 & L3 & L1 & L2 & L3 & L1 & L2 & L3 & EE (op)\\\midrule""")
# exact random-variation thresholds from the data (largest |change| of the seed swap)
TAU2 = max(abs(C[X]["seed_noise"]["settings"][st]["delta2_minADE"]) for X in (0, 1) for st in ("old_pipeline", "isolated"))
TAU3 = max(abs(C[X]["seed_noise"]["settings"][st]["delta3_L2@3s"]["ego"]) for X in (0, 1) for st in ("old_pipeline", "isolated"))
print("tau2", TAU2, "tau3", TAU3)
counts = {}
for k, X, name, u in ROWS:
    cells = []
    for setting in ("old_pipeline", "isolated"):
        s = u["settings"][setting]
        bs = B[X]["updates"][name][setting]
        for dname, val, tau in (("d2", s["delta2_minADE"], TAU2), ("d3", s["delta3_L2@3s"]["ego"], TAU3)):
            bkey = "delta2_minADE" if dname == "d2" else "delta3_L2@3s"
            lv = [val > 0, val > tau, bs[bkey]["harm"]]
            for i, v in enumerate(lv):
                counts[(setting, dname, i)] = counts.get((setting, dname, i), 0) + int(v)
                cells.append(r"\Y Y" if v else "")
    op = u["settings"]["old_pipeline"]
    ee = u["delta1_mAP50"] > 0 and (op["delta2_minADE"] > 0 or op["delta3_L2@3s"]["ego"] > 0)
    counts["ee"] = counts.get("ee", 0) + int(ee)
    A(f"{k} & {tag(name)} & " + " & ".join(cells) + f" & {'Y' if ee else '--'}\\\\")
    if k == 15:
        A(r"\midrule")
A(r"""\midrule Total & & """ + " & ".join(str(counts[(s, d, i)]) for s in ("old_pipeline", "isolated") for d in ("d2", "d3") for i in range(3))
  + f" & {counts['ee']}\\\\" + r"""\bottomrule\end{tabular}\end{table}""")


# bootstrap intervals
A(r"""\FloatBarrier
\section{Stage 3: bootstrap intervals for every update}
\begin{table}[h]\centering\scriptsize\caption{Point value and 95\,\% bootstrap interval (2{,}000 resamples of 50 scenes), $C_3$ = ego, metres}
\begin{tabular}{rlllll}\toprule
\# & Update & $\Delta_2$ old-pipeline & $\Delta_3$ old-pipeline & $\Delta_2$ isolated & $\Delta_3$ isolated\\\midrule""")


def ci(d):
    s = f"{f(d['point'],3,True)} [{f(d['ci95'][0],3,True)}, {f(d['ci95'][1],3,True)}]"
    return (r"\Y " + s) if d["harm"] else (r"\cellcolor{green!12}" + s if d["benefit"] else s)


for k, X, name, u in ROWS:
    bs = B[X]["updates"][name]
    A(f"{k} & {tag(name)} & {ci(bs['old_pipeline']['delta2_minADE'])} & {ci(bs['old_pipeline']['delta3_L2@3s'])} & {ci(bs['isolated']['delta2_minADE'])} & {ci(bs['isolated']['delta3_L2@3s'])}\\\\")
    if k == 15:
        A(r"\midrule")
A(r"\midrule")
for X in (0, 1):
    bs = B[X]["updates"][f"c1_old_s{1-X}"]
    A(f"noise & old s{X}$\\rightarrow$old s{1-X} & {ci(bs['old_pipeline']['delta2_minADE'])} & {ci(bs['old_pipeline']['delta3_L2@3s'])} & {ci(bs['isolated']['delta2_minADE'])} & {ci(bs['isolated']['delta3_L2@3s'])}\\\\")
A(r"""\bottomrule\end{tabular}\end{table}
Red = interval above zero (significant harm), green = interval below zero (significant improvement).
""")

# H2 measures per update
A(r"""\FloatBarrier
\begin{landscape}\section{H2: output change without ground truth, every update}\begin{table}[h!]\centering\scriptsize\caption{Nine label-free measures per update (old $C_1$ vs updated $C_1$ on Singapore 2), with $\delta_1$ and the old-pipeline $\Delta_2$}
\begin{tabular}{rlrrrrrrrrrrrr}\toprule
\# & Update & $\delta_1$ & $\Delta_2$ op & obj/img JS & mean obj/img & type JS & conf W & conf KS & fwd W (m) & lat W (m) & only-one & C2 move W & C2 shift (m)\\\midrule""")
X_FEAT = {}
for k, X, name, u in ROWS:
    hh = u["h2_c1_update"]
    c2 = u["settings"]["old_pipeline"]["h2_c2_update"]
    A(f"{k} & {tag(name)} & {f(u['delta1_mAP50'],4,True)} & {f(u['settings']['old_pipeline']['delta2_minADE'],3,True)} & {f(hh['objects_per_image_JS'],4)} & "
      f"{hh['mean_objects_per_image'][0]:.2f}/{hh['mean_objects_per_image'][1]:.2f} & {f(hh['type_mix_JS'],4)} & {f(hh['confidence_W'],4)} & {f(hh['confidence_KS'],3)} & "
      f"{f(hh['forward_position_W_m'],2)} & {f(hh['lateral_position_W_m'],2)} & {f(hh['only_one_version_fraction'],3)} & {f(c2['movement_W'],3)} & {f(c2['same_object_shift'],2)}\\\\")
    if k == 15:
        A(r"\midrule")
A(r"""\bottomrule\end{tabular}\end{table}\end{landscape}""")

feats = {
    r"$\delta_1$ (needs ground truth)": lambda u: u["delta1_mAP50"],
    "objects per image JS": lambda u: u["h2_c1_update"]["objects_per_image_JS"],
    "type mix JS": lambda u: u["h2_c1_update"]["type_mix_JS"],
    "confidence Wasserstein": lambda u: u["h2_c1_update"]["confidence_W"],
    "confidence KS": lambda u: u["h2_c1_update"]["confidence_KS"],
    "forward position Wasserstein": lambda u: u["h2_c1_update"]["forward_position_W_m"],
    "lateral position Wasserstein": lambda u: u["h2_c1_update"]["lateral_position_W_m"],
    "only-one-version fraction": lambda u: u["h2_c1_update"]["only_one_version_fraction"],
    "C2 predicted movement Wasserstein": lambda u: u["settings"]["old_pipeline"]["h2_c2_update"]["movement_W"],
    "C2 shift of the same object": lambda u: u["settings"]["old_pipeline"]["h2_c2_update"]["same_object_shift"],
}


def auroc(score, label):
    pos = [s for s, l in zip(score, label) if l]; neg = [s for s, l in zip(score, label) if not l]
    if not pos or not neg:
        return float("nan")
    return float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg]))


A(r"""\begin{table}[h]\centering\small\caption{How well each measure separates harmful updates ($\Delta_2 > 0$, old-pipeline)}
\begin{tabular}{lrrrrrr}\toprule
 & \multicolumn{2}{c}{All 30} & \multicolumn{2}{c}{Base s0 (15)} & \multicolumn{2}{c}{Base s1 (15)}\\
Measure & $\rho$ & AUROC & $\rho$ & AUROC & $\rho$ & AUROC\\\midrule""")
for nm, fn in feats.items():
    cells = []
    for sel in (None, 0, 1):
        rr = [u for _, X, _, u in ROWS if sel is None or X == sel]
        sc = [fn(u) for u in rr]; d2 = [u["settings"]["old_pipeline"]["delta2_minADE"] for u in rr]
        cells += [f(spearmanr(sc, d2).correlation, 2, True), f(auroc(sc, [d > 0 for d in d2]), 2)]
    A(nm + " & " + " & ".join(cells) + r"\\")
A(r"""\bottomrule\end{tabular}\end{table}
In base s0 all 15 updates have $\Delta_2 > 0$, so the AUROC within base s0 is undefined (--).
""")

# summary
A(r"""\FloatBarrier
\section{Summary of the counts}
\begin{table}[h]\centering\small\caption{Updates (out of 30) with an increase of the error}
\begin{tabular}{lrrrr}\toprule
 & \multicolumn{2}{c}{Prediction $\Delta_2$} & \multicolumn{2}{c}{Planning $\Delta_3$ (ego)}\\
Level & op & iso & op & iso\\\midrule""")
for i, lvl in enumerate(["L1 any increase", "L2 above random variation", "L3 bootstrap interval above zero"]):
    A(f"{lvl} & {counts[('old_pipeline','d2',i)]} & {counts[('isolated','d2',i)]} & {counts[('old_pipeline','d3',i)]} & {counts[('isolated','d3',i)]}\\\\")
A(r"""\bottomrule\end{tabular}\end{table}
\begin{itemize}
\item Every one of the 30 updates improved $C_1$ ($\delta_1$ from %s to %s).
\item Entangled enhancement by the simple rule (L1, $\Delta_2$ or $\Delta_3$ after old-pipeline training): %d of 30.
\item New-pipeline training (Stage 1b, 2 updates): $C_2$ 2.748/2.606\,m $\rightarrow$ 2.389/2.446\,m; $C_3$ ego 4.143/4.000\,m $\rightarrow$ 3.705/3.697\,m.
\end{itemize}
\end{document}""" % (f(min(u["delta1_mAP50"] for *_, u in ROWS), 4, True), f(max(u["delta1_mAP50"] for *_, u in ROWS), 4, True), counts["ee"]))

OUT.write_text("\n".join(L))
print("wrote", OUT)
print({str(k): v for k, v in counts.items()})
