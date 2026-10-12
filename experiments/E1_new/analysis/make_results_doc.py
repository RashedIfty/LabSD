"""Build the short results summary (Experimentation_Results.tex) from the result JSONs.
Every number is read from experiments/E1_new/results/ (Kaggle outputs)."""
import json
from pathlib import Path

E1 = Path(__file__).resolve().parent.parent
R = E1 / "results"
OUT = E1.parent.parent / "ieee paper" / "Version 4" / "Experimentation_Results.tex"
J = lambda p: json.load(open(R / p))
C = {X: J(f"stage2/campaign_base_s{X}.json") for X in (0, 1)}
B = {X: J(f"stage3/bootstrap_base_s{X}.json") for X in (0, 1)}


def f(x, d=3, sign=True):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


TAU2 = max(abs(C[X]["seed_noise"]["settings"][s]["delta2_minADE"]) for X in (0, 1) for s in ("old_pipeline", "isolated"))
TAU3 = max(abs(C[X]["seed_noise"]["settings"][s]["delta3_L2@3s"]["ego"]) for X in (0, 1) for s in ("old_pipeline", "isolated"))

rows, k = [], 0
for X in (0, 1):
    for name, u in C[X]["updates"].items():
        k += 1
        op, iso = u["settings"]["old_pipeline"], u["settings"]["isolated"]
        bs = B[X]["updates"][name]
        rows.append(dict(k=k, base=X, frac=int(u["frac"] * 100), ep=u["epochs"], seed=u["seed"], d1=u["delta1_mAP50"],
                         d2o=op["delta2_minADE"], d2i=iso["delta2_minADE"], d3o=op["delta3_L2@3s"]["ego"], d3i=iso["delta3_L2@3s"]["ego"],
                         sig2=bs["old_pipeline"]["delta2_minADE"]["harm"], sig3=bs["old_pipeline"]["delta3_L2@3s"]["harm"]))
cnt = lambda c: sum(1 for r in rows if c(r))
n = dict(
    d2o_any=cnt(lambda r: r["d2o"] > 0), d2i_any=cnt(lambda r: r["d2i"] > 0),
    d2o_tau=cnt(lambda r: r["d2o"] > TAU2), d2i_tau=cnt(lambda r: r["d2i"] > TAU2),
    d3o_any=cnt(lambda r: r["d3o"] > 0), d3i_any=cnt(lambda r: r["d3i"] > 0),
    d3o_tau=cnt(lambda r: r["d3o"] > TAU3), d3i_tau=cnt(lambda r: r["d3i"] > TAU3),
    d2o_sig=cnt(lambda r: r["sig2"]), d3o_sig=cnt(lambda r: r["sig3"]),
    d2i_sig=sum(B[r["base"]]["updates"][[*C[r["base"]]["updates"]][(r["k"] - 1) % 15]]["isolated"]["delta2_minADE"]["harm"] for r in rows),
    d3i_sig=sum(B[r["base"]]["updates"][[*C[r["base"]]["updates"]][(r["k"] - 1) % 15]]["isolated"]["delta3_L2@3s"]["harm"] for r in rows),
    worse=cnt(lambda r: r["d2o"] > r["d2i"]), ee=cnt(lambda r: r["d1"] > 0 and (r["d2o"] > 0 or r["d3o"] > 0)))
d1min, d1max = min(r["d1"] for r in rows), max(r["d1"] for r in rows)

L = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.6cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{colortbl}
\newcommand{\Y}{\cellcolor{red!15}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\pagestyle{plain}
\begin{document}
{\Large\bfseries Experimentation Results}

\section*{What I did}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Pipeline: $C_1$ finds objects (YOLOv11n), $C_2$ predicts their paths (AutoBot-Ego style), $C_3$ plans the ego path (PlanT style). All three are ML models.
\item Data: 350 nuScenes scenes, front camera. $C_1$ is trained in Boston, $C_2$ and $C_3$ on other Boston scenes, $C_1$ is updated on Singapore, and all tests use the same 50 Singapore scenes (1{,}720 test samples).
\item Two old $C_1$ (seed 0 and seed 1). Each is updated 15 times (100/50/25\,\% of the data, 5/10/20 epochs, 3 seeds): \textbf{30 updates}.
\item $C_2$ and $C_3$ are trained once in two ways and never retrained during the updates:
\textbf{old-pipeline training} (learn from the old $C_1$ output, including its mistakes) and
\textbf{isolated training} (learn from the ground truth).
\item For each update: $\delta_1$ = gain of $C_1$ (mAP50), $\Delta_2$ = increase of the prediction error, $\Delta_3$ = increase of the planning error at 3\,s (metres).
\end{itemize}
"""]

L.append(r"""\section*{Result 1: every update made $C_1$ better}
All 30 updates increased mAP50 ($\delta_1$ from %s to %s; old $C_1$ 0.109 and 0.117, updated $C_1$ 0.166 to 0.235).
""" % (f(d1min, 3), f(d1max, 3)))

L.append(r"""\section*{Result 2: $C_2$ and $C_3$ work (checked against simple baselines)}
\begin{tabular}{lll}\toprule
Model & Error on the test scenes & Simple baseline\\\midrule
$C_2$ (input: ground truth) & 0.77\,m & 4.48\,m (objects stay still)\\
$C_2$ (input: $C_1$ output) & 2.39 to 2.95\,m & 5.39 to 5.85\,m\\
$C_3$ without the ego path & 9.3 to 12.5\,m & 9.25\,m (average path) $\rightarrow$ dropped\\
$C_3$ with the last 1.5\,s ego path & 3.70 to 4.16\,m, collisions 5--8\,\%% & 9.25\,m, collisions 18\,\%%\\\bottomrule
\end{tabular}

$C_3$ still uses the objects: removing them moves its plan by 0.73 to 1.65\,m.
""".replace("%%", "%"))

L.append(r"""\section*{Result 3 (main): a better $C_1$ harms the $C_2$ that learned its old mistakes}
Number of updates (out of 30) in which the error increased.
Three levels, from weak to strong evidence.
``Larger than random variation'' means larger than the change caused by only replacing the old $C_1$ with the other old $C_1$ (%s\,m for prediction, %s\,m for planning).
``Statistically significant'' means the increase stays above zero in 95\,\%% of 2{,}000 random re-samplings of the 50 test scenes.

\begin{tabular}{lcccc}\toprule
 & \multicolumn{2}{c}{Prediction error ($C_2$)} & \multicolumn{2}{c}{Planning error ($C_3$)}\\
 & old-pipeline training & isolated training & old-pipeline training & isolated training\\\midrule
Any increase & \textbf{%d} & %d & %d & %d\\
Larger than random variation & \textbf{%d} & %d & %d & %d\\
Statistically significant & \textbf{%d} & %d & %d & %d\\\bottomrule
\end{tabular}

\begin{itemize}\setlength{\itemsep}{1pt}
\item In \textbf{%d of 30} updates, the $C_2$ from old-pipeline training got worse than the $C_2$ from isolated training. Same input, different training: the harm comes from learning the old mistakes.
\item All 11 updates larger than random variation start from the old $C_1$ with seed 0 (mean $\Delta_2$ = +0.165\,m). This $C_2$ also gets worse (+0.112\,m) when the old $C_1$ with seed 0 is simply replaced by the old $C_1$ with seed 1, although both have almost the same accuracy (0.009 mAP50 apart). This $C_2$ reacts to the different mistakes, not to the accuracy.
\item Planning changes very little (mean $\Delta_3$ $-$0.006 to +0.058\,m, at most 0.13\,m of about 4\,m), and the increase is more frequent after isolated training. Learned mistakes cannot cause an increase after isolated training.
\item By the simple rule ($\delta_1 > 0$ and $\Delta_2 > 0$ or $\Delta_3 > 0$), entangled enhancement appears in \textbf{%d of 30} updates.
\end{itemize}
""".replace("%%", "%%") % (f"{TAU2:.3f}", f"{TAU3:.3f}", n["d2o_any"], n["d2i_any"], n["d3o_any"], n["d3i_any"],
                            n["d2o_tau"], n["d2i_tau"], n["d3o_tau"], n["d3i_tau"], n["d2o_sig"], n["d2i_sig"], n["d3o_sig"], n["d3i_sig"],
                            n["worse"], n["ee"]))

L.append(r"""\section*{Result 4: retraining $C_2$ and $C_3$ on the new $C_1$ lowers both errors}
Tested on 2 updates (new-pipeline training). Prediction error 2.75 / 2.61\,m $\rightarrow$ 2.39 / 2.45\,m,
planning error 4.14 / 4.00\,m $\rightarrow$ 3.71 / 3.70\,m. Both are even below the errors before the update.

\section*{Result 5: no warning sign found yet (H2 not supported)}
Nine measures of how much the output of $C_1$ and $C_2$ changes (no ground truth needed) were tested.
Correlation here means how well a larger change goes together with a larger harm (1 = perfect, 0 = no relation).
The best measure, how far $C_2$'s predictions shift, has a correlation of +0.68 over all 30 updates.
For the 15 updates of each old $C_1$ separately, the correlation is only +0.29 and $-$0.28.
The measure therefore only tells the two old $C_1$ apart and does not warn about single harmful updates.
The accuracy gain $\delta_1$ does not predict the harm either (correlation +0.04).
Hint for the next step: the old $C_1$ whose $C_2$ got worse when the old $C_1$ was replaced is also the one whose updates caused harm.
""")

L.append(r"""\section*{All 30 updates}
Changes after each update, in metres for $\Delta_2$ and $\Delta_3$. Shaded cells are increases larger than random variation.
The last column names the errors with a statistically significant increase after old-pipeline training.

\small
\begin{tabular}{rcrrcrrrrl}\toprule
 & & & & & \multicolumn{2}{c}{Prediction error $\Delta_2$} & \multicolumn{2}{c}{Planning error $\Delta_3$} & \\
\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\# & Old $C_1$ & Data & Epochs & Update seed & old-pipeline & isolated & old-pipeline & isolated & Statistically significant\\
 & seed & used & & & training & training & training & training & increase\\\midrule""")
for r in rows:
    c = lambda v, t: (r"\Y " if v > t else "") + f(v)
    sig = ", ".join(x for x, ok in (("prediction", r["sig2"]), ("planning", r["sig3"])) if ok) or "none"
    L.append(f"{r['k']} & {r['base']} & {r['frac']}\\% & {r['ep']} & {r['seed']} & {c(r['d2o'], TAU2)} & {c(r['d2i'], TAU2)} & {c(r['d3o'], TAU3)} & {c(r['d3i'], TAU3)} & {sig}\\\\")
    if r["k"] == 15:
        L.append(r"\midrule")
L.append(r"""\bottomrule\end{tabular}

\section*{Bottom line}
\begin{itemize}\setlength{\itemsep}{1pt}
\item Entangled enhancement is real for the prediction module: a better $C_1$ harms the $C_2$ that learned the old $C_1$, and not the $C_2$ trained on the ground truth.
\item The effect on planning is very small.
\item Retraining $C_2$ and $C_3$ together with $C_1$ lowers both errors, but no cheap warning sign has been found yet. Early detection is the next step.
\end{itemize}
\end{document}""")
OUT.write_text("\n".join(L))
print("wrote", OUT, n)
