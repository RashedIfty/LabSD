# End-to-End vs. Pipeline Approaches and Baseline Methods for Mitigating Cascading Effects

**Rashedul Arefin Ifty — University of Tsukuba**
**Meeting 5 Report — Literature Review (Tasks assigned at Meeting 4)**
**July 2026**

---

## 1. Executive Summary

This report answers the two research tasks assigned at Meeting 4.

**Task 1 (E2E vs. pipeline).** End-to-end (E2E) driving models — UniAD, VAD, PARA-Drive, SparseDrive, Hydra-MDP, and the 2025–2026 wave of VLA-based models — are indeed the dominant recent research trend (270+ papers by 2024 per the TPAMI survey [6]). However, the literature itself documents that E2E models (i) are black boxes with unresolved interpretability, causal-confusion, and robustness problems [6][7], (ii) cannot be selectively retrained — repairing a failure requires fine-tuning the *entire* model, which causes catastrophic forgetting [13][14], and (iii) are evaluated with open-loop nuScenes metrics that the community has shown to be unreliable [9][10][11]. Most importantly for us: **our research object is the component-level retraining policy itself.** The SMP framework requires per-component binary states ⟨s1,s2,s3⟩, per-component metrics, and independently executable retraining actions (the 7 subsets E1–E7). A monolithic E2E model has *no component states, no component metrics, and exactly one retraining action* — the phenomenon we study (entangled enhancement under partial retraining) is structurally unobservable in an E2E model. The pipeline choice is therefore not a concession; it is the only architecture in which the research question is well-posed. Strikingly, the emerging E2E *repair* literature (E2EREP, CorrectAD, R2SE) is being forced to reintroduce component-level views and modular adapters into E2E models to make maintenance tractable — which validates, rather than undermines, our modular premise.

**Task 2 (cascade mitigation).** The literature offers three families of mitigation: (a) *detection before propagation* — drift/OOD monitoring on intermediate interfaces (TFDV-style two-sample tests, OOD detectors, conformal uncertainty propagation); (b) *regression-constrained retraining* — backward-compatible training (BCT), positive-congruent training (PC-training/focal distillation), ELODI; and (c) *architectural/process controls* — coordinated retraining, shadow deployment, contracts on module interfaces. **Recommended first baseline: a C1→C2 interface drift gate** — a two-sample distribution test (KS test / Jensen–Shannon divergence) comparing the old and new detector's output statistics on a fixed audit set, run *before* the retrained C1 is admitted into the pipeline. It is implementable in a day on the existing Kaggle pipeline, requires no new training, and produces exactly the early-warning signal a retraining policy in the SMP can condition on.

---

## 2. Literature Review of End-to-End Models

### 2.1 The flagship E2E architectures

**[1] UniAD — Hu et al., "Planning-oriented Autonomous Driving," CVPR 2023 (Best Paper Award).**
- **Main idea:** Unify all driving tasks (detection, tracking, mapping, motion forecasting, occupancy, planning) in one network, with every module designed to serve the final planning objective.
- **Methodology:** Query-based design: unified query interfaces connect BEV perception, prediction, and planning transformer modules; trained jointly end-to-end on nuScenes; evaluated open-loop (L2, collision rate).
- **Strengths:** First "planning-oriented" unified framework; intermediate queries give partial interpretability; strong reported open-loop numbers; hugely influential.
- **Weaknesses:** Very heavy: SparseDrive reports UniAD needs ~144 h training and runs at 1.8 FPS [4]; open-loop evaluation protocol later shown inconsistent [3][10]; joint training means any update touches everything.
- **Relevance:** The canonical E2E system our pipeline is contrasted against; also the primary experimental subject of the E2EREP repair paper the advisor shared.

**[2] VAD — Jiang et al., "VAD: Vectorized Scene Representation for Efficient Autonomous Driving," ICCV 2023.**
- **Main idea:** Replace dense rasterized BEV representations with fully vectorized scene representations (map polylines, agent vectors) to make E2E planning far more efficient.
- **Methodology:** Vectorized encoders for maps and agents feeding a planning transformer with explicit vectorized constraints (collision, boundary, direction); trained end-to-end on nuScenes.
- **Strengths:** Much faster than UniAD at comparable or better open-loop planning metrics; vectorized constraints add a measure of safety structure.
- **Weaknesses:** Same open-loop evaluation weaknesses; still a monolithic jointly-trained model; PARA-Drive showed the UniAD–VAD comparison was confounded by inconsistent evaluation protocols [3].
- **Relevance:** Second experimental subject of E2EREP; together UniAD/VAD define the current "repairable E2E model" testbed.

**[3] PARA-Drive — Weng et al. (NVIDIA), "PARA-Drive: Parallelized Architecture for Real-time Autonomous Driving," CVPR 2024.**
- **Main idea:** Systematically explore the E2E design space (which modules are necessary, where they sit, how information flows) and show that a *fully parallel* architecture — all heads co-trained off a shared BEV feature, no sequential perception→prediction→planning flow — matches or beats sequential E2E stacks.
- **Methodology:** Co-train mapping/motion/occupancy/planning heads in parallel on shared BEV features; heads can be *deactivated at runtime*; unified re-evaluation of UniAD/VAD under one protocol.
- **Strengths:** Up to 28.8% lower planning L2 and 43.3% lower collision rate vs. UniAD/VAD/OccNet; 2.77× runtime speedup; exposed that the published UniAD-vs-VAD gap largely evaporates under a unified protocol.
- **Weaknesses:** Open-loop nuScenes only (authors state closed-loop is future work); "parallel" heads still share one jointly-trained backbone — they are not independently retrainable.
- **Relevance:** Two lessons: (i) even within the E2E camp, *architecture* (module wiring) matters and is actively debated; (ii) published E2E open-loop comparisons are protocol-fragile — a caution for how much weight to give E2E leaderboard numbers.

**[4] SparseDrive — Sun et al., "SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation and Parallel Planning," arXiv:2405.19620, 2024.**
- **Main idea:** Fully sparse (object-centric) scene representation instead of dense BEV; symmetric sparse perception (detection+tracking+online mapping) and a parallel motion planner with collision-aware rescoring.
- **Methodology:** Sparse query-based perception; hierarchical planning selection; trained end-to-end on nuScenes.
- **Strengths:** Reports 0.58 m avg L2 / 0.06% collision (vs. VAD 0.72 m / 0.21%); **7.2× faster training (20 h vs 144 h) and 5× faster inference (9.0 vs 1.8 FPS) than UniAD** — a rare quantified admission of how expensive E2E (re)training is.
- **Weaknesses:** Open-loop only; frames the modular pipeline critique ("information loss, error accumulation") rhetorically, without maintenance-phase analysis.
- **Relevance:** Its own headline numbers are evidence for our *retraining-cost* argument: if one full training run of a flagship E2E model costs 144 GPU-hours, per-shift retraining policies of the kind we study are economically prohibitive in E2E form.

**[5] Hydra-MDP — Li et al. (NVIDIA), "Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation," arXiv:2406.06978, 2024. Winner, CVPR 2024 Autonomous Grand Challenge (NAVSIM track).**
- **Main idea:** Distill multiple teachers — human demonstrations *and* rule-based planners — into a multi-head student that scores trajectory candidates against multiple simulation-derived metrics.
- **Methodology:** Multi-target hydra-distillation; multi-head decoder over a fixed trajectory vocabulary; trained/evaluated on NAVSIM.
- **Strengths:** State of the art on NAVSIM; shows rule-based knowledge still carries much of the value (echoing PDM-Closed [11]).
- **Weaknesses:** Deliberately *internalizes* the rule-based module into the differentiable network — erasing the modular boundary and thus the ability to inspect or update the rule component separately.
- **Relevance:** Illustrates the trend of absorbing modular structure into monoliths; the cost is exactly the maintainability our project studies.

**[6] Survey — Chen, Wu, Chitta, Jaeger, Geiger, Li, "End-to-End Autonomous Driving: Challenges and Frontiers," IEEE TPAMI 2024 (arXiv:2306.16927).**
- **Main idea/scope:** Definitive survey of 270+ E2E driving papers; taxonomy of imitation vs. RL approaches; open challenges.
- **Key content for us:** Names **interpretability, causal confusion, and robustness (generalization under distribution shift)** as *unresolved* challenges; notes E2E growth is driven by large-scale data and closed-loop benchmarks (i.e., heavy data requirements); does **not** treat maintenance, retraining policy, or component-level reliability at all.
- **Strengths:** Authoritative, recent, peer-reviewed; written by E2E proponents (UniAD authors among them), so its admissions against interest are strong evidence.
- **Weaknesses:** As a survey, no new empirical results.
- **Relevance:** Primary citation for both the "E2E is the trend" premise and the unresolved-weaknesses list; its silence on retraining policy is our research gap, quotable directly.

**[7] Survey — "Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future," arXiv:2512.16760, 2025.**
- **Main idea/scope:** Surveys the 2024–2026 wave of VLA driving models; organizes them into End-to-End VLA (one model does perception+reasoning+planning) and Dual-System VLA (slow VLM deliberation + fast low-level policy).
- **Key content for us:** Even in the newest paradigm, **robustness, interpretability, and instruction fidelity remain open problems**, and *component-level retraining/maintenance policy is not a studied topic* — the gap persists into the VLA era. Notably, the Dual-System designs are themselves a partial retreat to modularity (two separately-designed subsystems).
- **Relevance:** Shows our gap is not closed by the latest model class; the trend line inside E2E research bends back toward structure.

### 2.2 The E2E evaluation critique (why E2E leaderboard numbers must be read carefully)

**[8] Codevilla et al., "On Offline Evaluation of Vision-based Driving Models," ECCV 2018.**
- **Main idea:** Offline (open-loop) prediction error is not necessarily correlated with actual closed-loop driving quality; two models with identical offline error can drive dramatically differently.
- **Strengths/weaknesses:** Foundational, widely replicated finding; older benchmark (CARLA v1) but conclusion repeatedly reconfirmed [9][10].
- **Relevance:** Both E2E models *and our C3 L2 metric* live in open-loop land; this is a caveat we must state for our own Table I as well — an honest limitation to carry in the thesis.

**[9] Zhai et al. (Baidu), "Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes," arXiv:2305.10430, 2023 (the "AD-MLP" paper).**
- **Main idea:** A trivial MLP using **only ego state (past trajectory, velocity), no perception at all**, matches or beats perception-based E2E methods on nuScenes open-loop L2 — exposing how weak the nuScenes open-loop planning protocol is (though perception-based methods keep an edge on collision rate).
- **Relevance:** Devastating for naive E2E-vs-E2E comparisons on nuScenes; also directly relevant to us since our C3 metric is nuScenes open-loop L2 — we should report collision rate alongside L2 (we already do) and frame L2 deltas comparatively (before vs. after), not absolutely.

**[10] Dauner, Hallgarten, Geiger, Chitta, "Parting with Misconceptions about Learning-based Vehicle Motion Planning," CoRL 2023.**
- **Main idea:** Open-loop ego-forecasting and closed-loop driving are misaligned tasks; a largely **rule-based planner (PDM-Closed) won the 2023 nuPlan challenge**, beating learned planners; on the open-loop sub-task, best results came from using *only the centerline* and ignoring all agents.
- **Relevance:** Triple duty for us: (i) justifies our rule-based IDM C3 (it reproduces the architecture of the strongest-known nuScenes-era planner class), (ii) is the strongest citation that "learned E2E planning is better" is unproven in closed loop, (iii) motivates our Tier-1/Tier-2 metric separation.

**[11] Dauner et al., "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking," NeurIPS 2024.**
- **Main idea:** Middle ground between open- and closed-loop: short-horizon non-reactive simulation yields metrics (PDM score: progress, time-to-collision, drivable area) far better aligned with closed-loop performance; validated by a 143-team CVPR 2024 competition.
- **Key finding:** Simple, moderate-compute E2E methods (TransFuser) **match** flagship architectures (UniAD) on challenging scenarios — architectural complexity in E2E driving buys less than leaderboards suggest.
- **Relevance:** If we later want a stronger C3 pipeline metric than open-loop L2, NAVSIM-style PDM scoring on nuScenes-mini scenes is the natural upgrade path (feasible: it needs only BEV abstractions, not a full simulator).

### 2.3 The emerging E2E *repair/maintenance* literature (the advisor's E2EREP thread)

**[12] E2EREP — component-level repair of E2E ADS (paper page shared by advisor; not yet publicly indexed — likely under review/unpublished; we should request the full PDF).**
- **Main idea (from the shared page):** Repairing an E2E model by evaluating it at the **"component level"** using cheap *open-loop* metrics (trajectory L2 deviation, predicted collisions), because closed-loop simulation of each candidate repair takes tens of minutes and full repair would take days.
- **Methodology (from the shared page):** Improved fault-localization splitting of positive/negative data by *impact on driving performance* (collision-prone inputs labeled negative) rather than a naive L2 median split; six E2EREP variants evaluated on **UniAD and VAD**.
- **Reported findings:** Their fault localization beats the baseline; repair consistently improves the model; **but better open-loop performance "does not always correspond to better closed-loop performance"** — their own closing caveat.
- **Relevance — this paper is a gift to our argument:** to make E2E models *maintainable*, the authors must (a) impose a virtual component decomposition on the monolith and (b) accept open-loop proxies with acknowledged validity limits. Both are things a modular pipeline gives natively and exactly. It also confirms "repair/maintenance of driving models" is an active SE research frontier adjacent to Prof. Machida's field.

**[13] CorrectAD, "A Self-Correcting Agentic System to Improve End-to-end Planning in Autonomous Driving," arXiv:2511.13297, 2025.**
- **Main idea:** Automated failure-repair loop for E2E planners: diagnose failure cases, generate targeted synthetic training video (diffusion model, "DriveSora"), then **fine-tune the entire E2E model** on old + new data.
- **Findings:** Corrects 62.5% of failure cases on nuScenes (49.8% in-house); collision rate −39%. Validated on UniAD and VAD.
- **Weaknesses:** Whole-model retraining per repair cycle; needs an expensive generative-video pipeline because long-tail failure data cannot be safely collected.
- **Relevance:** Direct evidence that in an E2E model **the smallest retraining unit is the whole model** — the 7-subset policy space we study collapses to a single action. Also evidence of the enormous per-repair cost.

**[14] R2SE, "Reinforced Refinement with Self-aware Expansion for End-to-End Autonomous Driving," arXiv:2506.09800, 2025.**
- **Main idea:** Avoid full-model retraining when fixing hard cases: freeze the generalist E2E model, train **low-rank residual adapter "specialists"** with RL on failure regions, merge back.
- **Findings:** Improves Hydra-MDP from 87.0 → 91.6 PDMS on NAVSIM; explicitly motivated by the observation that **naive fine-tuning of an E2E model on failure cases causes catastrophic forgetting** of general driving skill.
- **Relevance:** The E2E community's own workaround for un-retrainability is to bolt *modular, separately-trained parts* onto the monolith. (See also ADReFT, arXiv:2506.23960, 2025, which repairs at the decision/output level with an add-on transformer rather than touching the base model — same pattern.) Modularity is being rediscovered as the price of maintainability.

---

## 3. End-to-End vs. Pipeline Comparison

| Dimension | Modular pipeline (ours) | End-to-end | Evidence |
|---|---|---|---|
| **Interpretability** | Each interface (detections, trajectories) is human-inspectable; failure attributable to a component | Black box; interpretability an open challenge even per E2E surveys | [6][7]; Meeting 2 three-tier framework |
| **Modularity / selective update** | Any of 2³−1 = 7 retraining subsets executable; components swappable | One retraining action: the whole model; adapters (R2SE) are a partial patch | [13][14]; our E1 |
| **Ease of debugging** | Isolated vs. pipeline evaluation localizes faults (our Table I isolation property held bit-exactly) | Requires research-grade fault localization (E2EREP) with open-loop proxies of limited validity | [12]; Meeting 4 §6 |
| **Retraining / maintenance cost** | Retrain only the shifted component: our C1 fine-tune ≈ 50 s on a T4 | Full retrain: UniAD ≈ 144 GPU-h; repair loops need synthetic-data factories | [4][13]; Meeting 4 §5 |
| **Robustness under domain shift** | Shift effects measurable & attributable per component; rule-based C3 degrades predictably | Distribution-shift sensitivity and causal confusion unresolved; forgetting under fine-tuning | [6][7][14] |
| **Scalability (system growth)** | Add/replace components independently; SMP states scale 2^N | Any capability change = new full training run on full data | [4][6] |
| **Computational efficiency** | Light components run on free-tier hardware (our whole E1 fits Kaggle) | 1.8–9 FPS inference, multi-day training for flagship models | [3][4]; Meeting 4 §4 |
| **Data requirements** | Each component trainable on task-specific labels; small-data fine-tunes possible (with known overfitting caveats) | Growth driven by large-scale fleets/datasets; long-tail failure data unsafe to collect | [6][13] |
| **System-level optimality** | ✗ Modules optimize misaligned local objectives; information loss at interfaces — *this is the E2E camp's valid critique, and it is exactly the phenomenon our SMP models* | ✓ Joint optimization for planning | [1][4][6] |
| **Evaluation trustworthiness** | Per-component ground-truth metrics + isolation tests | Open-loop planning metrics on nuScenes shown unreliable (ego-only MLP wins) | [8][9][10][11] |

The last two rows are deliberately honest: E2E's genuine advantage is joint optimization, and the pipeline's genuine weakness — inter-component coupling loss — is precisely our research object, not a nuisance we ignore.

---

## 4. Why a Pipeline Approach Is Justified for This Research

**Argument 1 — The research question is only well-posed in a modular system.** Our SMP model requires per-component binary states ⟨s1,s2,s3⟩ determined by per-component metrics (mAP, minADE, L2 — Meeting 2, Table VIII), and its transitions are *retraining actions on component subsets* (E1–E7). An E2E model has no component states (nothing to threshold), no isolated component metrics (E2EREP must invent proxy ones [12]), and a one-element action space (CorrectAD fine-tunes everything [13]). One cannot study "which component to retrain, and in what order" in an architecture where that question cannot be asked. This is analogous to reliability engineering generally: fault-tree and Markov availability analysis presuppose an architecture with distinguishable failure units.

**Argument 2 — Entangled enhancement is a *systems* phenomenon that E2E hides, not solves.** The E2E camp's critique of pipelines — "errors compound across modules" [4][6] — concedes our premise: cascading effects are real, important, and architectural. E2E responds by making the cascade invisible (one loss, one model); we respond by making it *measurable* (isolated-vs-pipeline gap; ρ₁→₃). For a software-reliability research agenda, measurability wins: Sculley et al.'s CACE principle ("Changing Anything Changes Everything") and correction cascades [15] show the entanglement persists inside monoliths — negative flips occur at 4–5% even between equal-accuracy retrains of the *same* architecture [18] — it just stops being observable or attributable.

**Argument 3 — Maintenance economics.** Real MLS maintenance is a *repeated* activity under drift. Our per-component fine-tune costs ~50 GPU-seconds; a UniAD-class retrain costs ~144 GPU-hours [4], and E2E repair pipelines additionally require synthetic-data generation infrastructure [13] because failure data cannot be safely collected. A retraining *policy* study — dozens of retraining events across E1–E7, multiple seeds — is only executable, by roughly four orders of magnitude, in the modular setting. This mirrors industry practice: production AV stacks (Waymo, Mobileye, and the Boeing runway-incursion system analyzed in [22]) remain modular in large part for exactly these validation and maintenance reasons.

**Argument 4 — The E2E field itself is converging back toward modularity when maintenance matters.** E2EREP imposes component-level views on UniAD/VAD to localize faults [12]; R2SE freezes the monolith and trains modular adapters [14]; ADReFT adds a separate repair module; Dual-System VLA architectures split deliberation from control [7]; PARA-Drive makes heads deactivatable [3]. The direction of travel under maintenance pressure is monolith → structure. Our work sits where that trend is heading, with the added rigor of an explicit stochastic model.

**Argument 5 — Evaluation integrity.** E2E superiority claims rest heavily on nuScenes open-loop metrics that the community has discredited for *ranking* models [8][9][10]. Our use of the same open-loop L2 is defensible precisely because we use it *differentially* (same planner, same scenes, before/after one upstream change) rather than to rank architectures — but this limitation must be stated, and NAVSIM-style PDM scoring [11] is the identified upgrade path.

**Research gaps this framing exposes (thesis material):**
1. No published work studies *retraining policies* (what to retrain, when, in what order) for E2E driving models — the TPAMI and VLA surveys do not even index the topic [6][7].
2. The E2E repair literature is nascent (2025–2026, largely preprints) and evaluates repairs with open-loop proxies of acknowledged limited validity [12][13][14].
3. No one has connected the availability/continuity formalism of software reliability engineering to E2E model maintenance — our SMP extension is positioned exactly in that gap.

---

## 5. Literature Review on Cascading Effect Mitigation

### 5.1 Foundational systems view

**[15] Sculley et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS 2015.**
- **Key ideas:** *CACE* — no ML input signals are ever independent; changing anything changes everything. *Correction cascades* — models trained on other models' outputs create an "improvement deadlock" where improving any single component degrades the system. *Unstable data dependencies* — even genuine upstream improvements can arbitrarily harm consumers. Prescribes monitoring prediction bias and SLOs on upstream producers.
- **Strengths:** The canonical industrial statement of our phenomenon, ten years before our measurement; universally cited.
- **Limitations:** Diagnostic, not algorithmic — names the disease, offers process-level treatments only.
- **Relevance:** Our E1 result (C2 pipeline minADE −74% while C3 L2 +4.45%) is a controlled, quantified instance of the improvement-deadlock pattern; cite as the industrial motivation.

### 5.2 Family (a): Detect the cascade before it propagates

**[16] TFDV / Breck et al., "Data Validation for Machine Learning," MLSys 2019 (+ TFX documentation).**
- **Key idea:** Validate data flowing between pipeline stages against a (learned) schema; detect drift between data spans via L∞ distance (categorical) and approximate Jensen–Shannon divergence (numeric); detect training–serving skew. Validation is applied at *multiple intermediate points* of the pipeline.
- **Strengths:** Industrially proven; extremely cheap; no model changes.
- **Limitations:** Data-level only — flags distribution change, not task-metric regression; thresholds need tuning.
- **Difficulty: EASY.** This is the template for our recommended baseline (Section 6).

**[17] Nitsch et al., "Out-of-Distribution Detection for Automotive Perception," IEEE ITSC 2021.**
- **Key idea:** OOD detection on the perception component, requiring no OOD training data and no inference-time cost; raw softmax confidence is shown to be unreliable (high confidence on OOD inputs). OOD flags can trigger a safe fallback mode.
- **Strengths:** Component-level monitoring as a safety mechanism; automotive-specific.
- **Limitations:** Detects input-domain novelty, not output-interface shift after retraining; per-component design effort.
- **Difficulty: MEDIUM.**

**[18–20] Uncertainty propagation across module boundaries.**
- **[18] Ivanovic et al., "Propagating State Uncertainty Through Trajectory Forecasting," ICRA 2022:** passing only point estimates across the perception→prediction interface makes forecasts systematically overconfident; propagating state uncertainty through a CenterPoint + AB3DMOT + Trajectron++ stack **on nuScenes** significantly improves FDE. The standard "certainty-equivalent" module interface is named as partially responsible for real AV failures.
- **[19] MOT-CUP, IEEE RA-L 2024:** conformal prediction quantifies detector uncertainty and propagates it into tracking; +2% accuracy, 2.67× uncertainty reduction, benefits largest under occlusion; detector/tracker-agnostic add-on.
- **[20] Shao et al., arXiv:2403.02297, 2024 + system-level UQ analysis, arXiv:2410.12019, 2024:** planners that ignore prediction uncertainty have lower success rates; module uncertainty *cannot be evaluated in isolation* — its value depends on the downstream consumer; proposes assume–guarantee contracts derived backward from a system-level spec (a formal-methods bridge highly compatible with the SMP worldview). Also reports downstream modules can *absorb* rather than amplify upstream error — the mechanism that plausibly explains our non-monotone C2↑/C3↓ result.
- **Strengths:** Richer interfaces demonstrably reduce cascade damage; [18] is nearly our exact pipeline.
- **Limitations:** Requires modifying component interfaces and retraining consumers; research-grade.
- **Difficulty: ADVANCED** (but [18] is the natural "next experiment" after the baseline).

### 5.3 Family (b): Retrain in a regression-constrained way

**[21] Shen et al., "Towards Backward-Compatible Representation Learning" (BCT), CVPR 2020 (+ MixBCT arXiv:2308.06948, 2023; FastFill, ICLR 2023).**
- **Key idea:** Train the *new* version of an upstream model under an "influence loss" that keeps its outputs usable by unchanged downstream consumers — avoiding the need to reprocess/retrain downstream. Without BCT, an independently retrained embedding model is 0% compatible with its predecessor. FastFill notes backfilling after updates can take *months* in production — retraining cost is the blocker for updates at scale.
- **Strengths:** Directly targets our scenario (update C1, freeze C2/C3); mature line of work with successors.
- **Limitations:** Compatibility constrains the new model's own gains (explicit trade-off); designed for embedding/retrieval — mapping to detection outputs requires adaptation.
- **Difficulty: MEDIUM.**

**[22] Yan et al., "Positive-Congruent Training: Towards Regression-Free Model Updates," CVPR 2021 (+ ELODI, arXiv:2205.06265, 2022).**
- **Key idea:** Model updates cause *negative flips* — samples the old model got right that the new one gets wrong — even when aggregate accuracy improves; NFR is ~4–5% even between equal-accuracy same-architecture retrains. *Focal distillation* (upweight distillation loss on samples the old model handled correctly) reduces NFR while improving accuracy; ELODI's ensemble-distillation variant cuts NFR ~29% further.
- **Strengths:** Simple training-loss modification; conceptually the per-sample twin of entangled enhancement (aggregate metric up, behavior regressed).
- **Limitations:** Formulated for classification; needs adaptation to detection (per-box flips); reduces but does not eliminate output shift.
- **Difficulty: MEDIUM — the natural *second* baseline** (a PC-trained C1 fine-tune is a drop-in change to our YOLO training step).

### 5.4 Family (c): Process- and architecture-level controls

- **Coordinated/staged retraining** (retrain C2 on the new C1's outputs before deployment) — the Meeting 2 policy space already contains these as {C1,C2}-type actions; the literature analogue is "team retraining" from our own initial proposal, and Sculley's prescription of SLO'd producers [15]. **EASY–MEDIUM**, but multiplies retraining cost — the SMP is precisely the tool to decide *when* it is worth it.
- **Shadow / gated deployment** (run old and new C1 side-by-side on live traffic, admit the new one only if system-level metrics don't regress) — standard MLOps practice [16]; our isolated-vs-pipeline Table I *is* a gated evaluation done manually. **EASY** conceptually; the gate criterion is the research question.
- **Ensembling old+new components** — smooths output shift at 2× inference cost [22]. **EASY** but unprincipled for safety metrics.

---

## 6. Candidate Baseline Methods

| # | Method | Family | Difficulty | Verdict |
|---|---|---|---|---|
| B1 | **C1→C2 interface drift gate (two-sample tests on detector output statistics)** | detect | **Easy (~1 day)** | **★ Recommended first** |
| B2 | Positive-congruent (focal-distillation) fine-tuning of C1 | constrain retraining | Medium (~1 week) | Second |
| B3 | Coordinated retraining {C1,C2} as a policy arm | process | Easy–Medium | Already in E2–E7 plan |
| B4 | Uncertainty propagation across C1→C2 (Ivanovic-style) | detect/interface | Advanced | Later; needs interface redesign |
| B5 | OOD detector on C1 inputs | detect | Medium | Later; detects domain novelty, not retrain shift |
| B6 | NAVSIM-style PDM score for C3 | evaluation | Medium | Metric upgrade, not mitigation |

### The recommended baseline: B1 — Interface Drift Gate

**What it is.** Before admitting a retrained C1 into the pipeline, compare the *statistical profile of its outputs* against the incumbent C1's outputs on the same fixed audit set (e.g., all train+val keyframes): per-image detection count, confidence distribution, class histogram, box-center spatial distribution, and box-size distribution. Score each with a two-sample test — KS statistic for continuous features, Jensen–Shannon divergence for categorical — exactly the TFDV recipe [16] applied to an intermediate interface instead of input data. If any statistic exceeds a threshold τ_drift, flag "cascade risk: downstream consumers see a shifted input distribution" and route to the conservative policy arm (hold deployment, or trigger coordinated retraining) instead of blind promotion.

**Why this one first (justification):**
1. **Implementable now, at zero training cost.** It is ~100 lines of NumPy/SciPy over detection JSONs the Kaggle pipeline already emits; no new models, no GPU time, fits the free-tier constraint that governs everything else in E1.
2. **It would have fired on our real E1 data.** The Boston→Singapore fine-tune changed recall from 0.087 to 0.048 and mAP@50 from 0.119 to 0.064 — the detection-count and confidence profiles demonstrably shifted, which is exactly what the gate measures. We can validate the baseline *retrospectively* on the v22 artifacts before running anything new.
3. **It plugs directly into the SMP.** The gate's binary output is a *pre-transition observable*: a cheap predictor of whether the C1-retrain transition will land in a degraded-system state. That is precisely the "coverage factor" structure (c1, c_cas) of the model — the baseline is not a detour; it generates the data that calibrates the theory. It is also the simplest concrete instance of the "Entanglement Forecaster" from our initial proposal.
4. **Literature-grounded at every step.** Interface-level distribution monitoring is the prescribed industrial control for unstable data dependencies [15][16], and output-shift-despite-better-aggregate-metrics is the documented failure mode it catches [22].
5. **It respects the advisor's guidance** that simple baselines are acceptable, while leaving a clear escalation path: B2 (PC-training) *reduces* the shift the gate *detects*, giving a detect-then-mitigate pair for a future meeting.

**Limitations to state honestly.** The gate detects *distribution* shift, not *harm* — a shift can be benign (our C2 minADE improved!). Threshold τ_drift must be tuned, and with n=3 validation scenes, false-positive/negative rates cannot be estimated tightly. These limitations are themselves measurable with the E1 infrastructure, which makes the baseline a genuine experiment rather than a checkbox.

---

## References

[1] Y. Hu et al., "Planning-oriented Autonomous Driving," CVPR 2023 (Best Paper).
[2] B. Jiang et al., "VAD: Vectorized Scene Representation for Efficient Autonomous Driving," ICCV 2023.
[3] X. Weng et al., "PARA-Drive: Parallelized Architecture for Real-time Autonomous Driving," CVPR 2024.
[4] W. Sun et al., "SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation and Parallel Planning," arXiv:2405.19620, 2024.
[5] Z. Li et al., "Hydra-MDP: End-to-end Multimodal Planning with Multi-target Hydra-Distillation," arXiv:2406.06978, 2024.
[6] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, H. Li, "End-to-End Autonomous Driving: Challenges and Frontiers," IEEE TPAMI, 2024.
[7] "Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future," arXiv:2512.16760, 2025.
[8] F. Codevilla et al., "On Offline Evaluation of Vision-based Driving Models," ECCV 2018.
[9] J.-T. Zhai et al., "Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes," arXiv:2305.10430, 2023.
[10] D. Dauner, M. Hallgarten, A. Geiger, K. Chitta, "Parting with Misconceptions about Learning-based Vehicle Motion Planning," CoRL 2023.
[11] D. Dauner et al., "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking," NeurIPS 2024.
[12] "E2EREP" — component-level repair of E2E ADS with open-loop metrics (UniAD, VAD); page shared by advisor; not publicly indexed as of July 2026 — full citation pending (request PDF from advisor).
[13] "CorrectAD: A Self-Correcting Agentic System to Improve End-to-end Planning in Autonomous Driving," arXiv:2511.13297, 2025.
[14] "R2SE: Reinforced Refinement with Self-aware Expansion for End-to-End Autonomous Driving," arXiv:2506.09800, 2025. (See also ADReFT, arXiv:2506.23960, 2025.)
[15] D. Sculley et al., "Hidden Technical Debt in Machine Learning Systems," NeurIPS 2015.
[16] E. Breck et al., "Data Validation for Machine Learning," MLSys 2019; TensorFlow Data Validation documentation.
[17] J. Nitsch et al., "Out-of-Distribution Detection for Automotive Perception," IEEE ITSC 2021.
[18] B. Ivanovic et al., "Propagating State Uncertainty Through Trajectory Forecasting," ICRA 2022.
[19] S. Su et al., "Collaborative Multi-Object Tracking with Conformal Uncertainty Propagation," IEEE RA-L 2024.
[20] W. Shao et al., "Uncertainty-Aware Prediction and Application in Planning for Autonomous Driving," arXiv:2403.02297, 2024; "System-Level Analysis of Module Uncertainty Quantification in the Autonomy Pipeline," arXiv:2410.12019, 2024.
[21] Y. Shen et al., "Towards Backward-Compatible Representation Learning," CVPR 2020; "MixBCT," arXiv:2308.06948, 2023; "FastFill," ICLR 2023.
[22] S. Yan et al., "Positive-Congruent Training: Towards Regression-Free Model Updates," CVPR 2021; "ELODI: Ensemble Logit Difference Inhibition," arXiv:2205.06265, 2022.
[23] Z. Wang and F. Machida, "Exploiting the Availability–Continuity Trade-off in Imperfect Retraining of Machine Learning Systems," ISSRE 2025. (Base paper.)
