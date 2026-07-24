# Execution Plan — Responding to Sensei's Feedback on the E1 Paper

**Context:** Sensei reviewed the IEEE paper (`ieee paper/IEEE_Conference_Paper_E1.tex`,
"Entangled Enhancement in Modular Autonomous Driving Systems"). This document
analyzes the feedback and lays out an execution plan.

---

## 0. Phase 0 — Decisions locked (2026-07-23)

Student-side decisions made; sensei-side items still pending (see §8).

| Decision | Choice |
|---|---|
| **Paper direction** | **Target A (technical paper), fall back to B.** Do Q1 campaign first (serves both); attempt CARA evaluation (Q2); pick A if convincing, else B. |
| **Compute** | **Kaggle free tier only** (30 GPU-hr/week, T4). No full-nuScenes, no HPC, no NAVSIM-grade terminal metric this round. |
| **Q1 campaign scope** | **Minimal-convincing, ~18 runs:** 3 dataset sizes × 3 seeds (=9) + epochs sweep 3 configs × 3 seeds (=9). |
| **Venue / page limit** | **Same 8-page IEEEtran conference.** New results must fit; use appendix/trimming as needed. |

**Consequences of these choices for later phases:**
- Q1 (§3): drop the Singapore→Boston reverse-shift and the LR sweep from the
  *first* campaign (compute-limited); keep dataset-size axis + epochs axis +
  seeds. Reverse-shift stays a stretch goal if time allows.
- Q2 (§4): CARA evaluation is explicitly **proof-of-concept on n=3**; the
  drift-gate retrospective is the primary evidence, the full admission-rule
  check is secondary.
- Structure (§5): prepare for **Option A ordering** (CARA promoted before
  results, Related Work to back), but keep the B-compatible content ready.
- Page budget: 8-page ceiling means the campaign matrix likely goes to a
  compact table (+ possible appendix); plan figures accordingly.

**Still pending (sensei) — do not block Phase 1 on these:**
- Is n=3 acceptable for the CARA evaluation as proof-of-concept? (§8 Q4)
- Final confirmation of "target A, fall back to B" at sensei level. (§8 Q1)

> Phase 0 complete. **Next: Phase 1 (Q1 campaign experiments).**

---

## 1. Summary of the feedback

Sensei raised **two experiment questions**, restated the **core story**, offered
**two structural options** for the paper, and gave **two writing comments**.

### 1.1 Experiment questions
- **Q1 — Diversity/generality of fine-tuning.** The current paper reports a
  *single* fine-tuning scenario. Sensei wants multiple tuning campaigns (varying
  dataset size, tuning hyper-parameters, etc.) to test whether different
  entanglements appear. Showing *when/how* entanglement changes across campaigns
  would be an important conclusion.
- **Q2 — Evaluate the proposed method (CARA).** CARA is currently only an *idea*
  (a protocol + algorithm), with no empirical evaluation. Sensei wants at least
  some demonstration of its effectiveness.

### 1.2 The core story (must be delivered clearly)
1. Entangled enhancement is a *known* reliability concern in ML systems, but
   **no real entangled enhancement in an AV pipeline has been studied before.**
2. This paper **empirically shows real entangled enhancement induced by domain
   shift** in a perception–prediction–planning AV pipeline.
3. Based on the findings, we **propose a retraining assessment method** for safe
   ML-system maintenance.

### 1.3 Two structural options
- **(A) Technical research paper** — *if* we add CARA evaluation. More impactful.
  Structure: Intro → System model (incl. definition of entangled enhancement) →
  CARA → Experiment setup → Evaluation results → Related work → Conclusion.
- **(B) Practical experience report** — closer to the current version, focused on
  empirical findings. Structure: Intro → System model → Experiment setup →
  Results → **Practical implications** (multi-angle mitigation advice for
  practitioners, beyond CARA) → Related work → Conclusion.

### 1.4 Writing comments
- **W1 — Overloaded sentences.** Avoid abusing `:` and `;` to chain multiple
  topics. Split into shorter, elaborated sentences.
  - Example flagged: *"The focus is on C1, the retrained unit, and C2, its
    direct consumer, whose input interface carries the cascade; C3 gives the
    system-level reading."*
- **W2 — High-context / hard-to-interpret sentences.** Need more introductory
  explanation before terse claims.
  - Examples flagged: *"This paper supplies that measurement."*, *"The main
    instrument is dual-mode evaluation:"*, *"The gap between the modes isolates
    the cascade contribution…"*, *"The harness computes the five-quantity
    profile of Section III-D…"*, *"The three isolated-mode rows are exactly
    zero-delta."*, *"Table I exhibits a second inversion."*

---

## 2. Decision needed first: which structure (A or B)?

This is the **fork that governs everything else**, so it must be decided before
execution. The choice hinges on whether we can produce a credible **CARA
evaluation (Q2)** in the available time/compute.

| | (A) Technical paper | (B) Experience report |
|---|---|---|
| Requires | CARA implemented + evaluated | Multi-angle practical implications |
| Extra work | High (build gate, run it, measure TP/FP) | Moderate (analysis + writing) |
| Impact | Higher (sensei's words) | Solid, lower risk |
| Risk | CARA eval may be weak on n=3 scenes | Fewer new results needed |
| Compute | More Kaggle runs | Fewer runs |

**Recommendation:** Pursue **(A) as the target**, but structure the work so the
deliverables **degrade gracefully to (B)** if the CARA evaluation turns out too
thin. Concretely: do the multi-campaign experiments (Q1) *first* — they serve
**both** structures — then attempt the CARA evaluation (Q2). If Q2 is convincing,
go with (A); if not, the same results support (B) with a richer "Practical
implications" section.

> **ACTION 0 (decision):** Confirm with sensei that "target A, fall back to B" is
> acceptable, or get an explicit pick. Do this before heavy compute.

---

## 3. Experiment plan for Q1 (diversity of fine-tuning campaigns)

**Goal:** Move from one fine-tuning scenario to a *campaign matrix*, and report
how the entanglement pattern (signs of Δ₂, Δ₃, δ₁, and ρ) varies.

### 3.1 What is already parametrizable (feasibility check — confirmed)
The existing harness (`experiments/E1_kaggle/src/`) already supports this with
minimal changes:
- `c1_yolo.fine_tune_yolo(..., epochs, imgsz, batch)` — tuning knobs exposed.
- `build_yolo_dataset(...)` — dataset is built per scene-list, so **dataset size
  is controllable** by subsampling scenes/images.
- `eval.run_all_measurements(...)` returns the 5-quantity profile **plus
  per-scene** breakdowns; `table.cascade_diagnostic(...)` computes ρ.
- So the campaign matrix is mostly a **loop over configs**, not new algorithms.

### 3.2 Proposed campaign matrix (factors to vary)
Pick 2–3 factors, each with 2–3 levels, to keep the run count feasible on Kaggle:

1. **Fine-tune dataset size** (the most important for "generality"):
   - full Singapore train (119 imgs) / half (~60) / quarter (~30).
2. **Training length / capacity of the tune:**
   - epochs ∈ {5, 10, 20}; optionally learning-rate ∈ {low, default, high}.
3. **Domain-shift direction (nice-to-have, strengthens generality):**
   - Boston→Singapore (current) **and** Singapore→Boston (reverse).
4. **Random seed** (for the variance question sensei implicitly cares about):
   - ≥3 seeds per chosen config so we can report mean ± std, not single points.

**Suggested minimal-but-convincing set:** 3 dataset sizes × 3 seeds (=9 runs)
for the primary axis, plus a small epochs sweep (3 configs × 3 seeds = 9) →
~18 runs. Each C1 fine-tune ≈ 50 s on a T4; the whole matrix fits the free tier.

### 3.3 What to measure/report per campaign
For every config, record: δ₁ (C1 mAP change), Δ₂ (C2 pipe change), Δ₃ (C3 pipe
change), ρ₁→₃, collision rate, and the **regime label**:
- *entangled enhancement* (δ₁>0, Δ₃>0),
- *non-monotone* (Δ₂<0 improving while Δ₃>0),
- *metric decoupling* (δ₁<0 while Δ₂ improves) — the current paper's case,
- *clean/benign* (everything improves or unchanged).

**Deliverable:** a table + a scatter/heatmap of (δ₁, Δ₃) or ρ across the matrix,
showing which configs produce which regime. This directly answers Q1: "another
tuning campaign may exhibit different entanglements."

### 3.4 Honesty guardrail
n=3 Singapore validation scenes is still small. Report mean ± std over seeds and
state the confidence limits plainly. If a config's regime flips only within
seed variance, say so — that itself is a finding about fragility.

---

## 4. Experiment plan for Q2 (evaluate CARA)

**Goal:** Turn CARA from "an idea" into "an idea with evidence." CARA has two
components to evaluate separately.

### 4.1 The drift-gate front-end (the cheap, concrete part)
This is the same gate proposed as the "first baseline" in the Meeting 5 report
(KS test + Jensen–Shannon divergence on C1 output statistics). Evaluating it:
- **Implement** `drift_gate.py` over the detection JSONs the pipeline already
  emits (per-image count, confidence dist., class histogram, box-center/size).
- **Retrospective evaluation** on the campaign matrix from §3: for each config,
  does the gate's "cascade-risk" flag correlate with the actual sign/magnitude
  of Δ₃? Report as a small confusion matrix (gate-fires vs. system-regressed).
- **Threshold study:** sweep τ_drift and plot how the flag rate and the
  TP/FP trade-off move — this is the honest way to show the gate does something.

### 4.2 The full CARA admission rule (the harder part)
CARA admits/holds an update based on the terminal metric Δₙ. To *evaluate* it we
need updates with known "good/bad" ground truth — which is exactly what the §3
campaign matrix provides (multiple updates, each with a measured Δ₃).
- **Metric:** across all campaign updates, does CARA's admit/hold decision match
  the desired outcome? Because Δ₃ is open-loop L2 (which the paper itself
  flags as possibly rewarding "blindness"), also report the collision-rate-aware
  variant of the rule and discuss the tension (already noted in the paper's
  Discussion).
- **Realistic scope:** with n=3 scenes and ~18 updates, this is a *proof of
  concept*, not a full statistical validation. Frame it that way.

### 4.3 Fallback if Q2 stays thin
If the CARA evaluation is unconvincing, keep CARA as a *proposal* and pivot the
paper to **structure (B)**, expanding §5 "Practical implications" with
multi-angle mitigation advice (drift gate, coordinated retraining, PC-training,
BCT, uncertainty propagation, shadow/gated deploy, NAVSIM-grade terminal metric)
— all already surveyed in the Meeting 5 literature review, so no new lit search.

---

## 5. Restructuring plan (paper edits)

Regardless of A vs. B, several structural moves are needed. Map from **current**
sections to **target**.

### 5.1 Current structure (for reference)
Intro → Related Work → System Model & Methodology → Experimental Setup →
Results → Discussion → Proposed Method (CARA) → Conclusion.

### 5.2 Target — Option A (technical paper)
1. **Introduction** — reframe around the 3 core-story points: (i) entangled
   enhancement as a known ML reliability concern, (ii) the AV pipeline, (iii)
   the gap = *no real AV-pipeline study*, (iv) this paper = evidence + solution.
2. **System model** — **move the definition of entangled enhancement here**
   (currently split between Intro and the Cascade-Diagnostic subsection). Define
   the pipeline, dual-mode evaluation, cascade gap/injection, and the formal
   entangled-enhancement condition in one place.
3. **CARA (proposed technique)** — promote CARA to a first-class section *before*
   results (sensei's Option A puts the technique before the experiment).
4. **Experiment setup** — datasets, campaign matrix (§3), harness, metrics.
5. **Evaluation results** — both the empirical entanglement findings (§3) *and*
   the CARA evaluation (§4).
6. **Related work** — move to the back (sensei's Option A ordering).
7. **Conclusion.**

### 5.3 Target — Option B (experience report)
1. Introduction (same reframing as A).
2. System model (same; include the definition).
3. Experiment setup (campaign matrix).
4. Results.
5. **Practical implications** — NEW section: multi-angle mitigation advice for
   practitioners; CARA becomes one recommendation among several.
6. Related work.
7. Conclusion.

### 5.4 Notes common to both
- **Move Related Work to the back** in both options (currently §II).
- **Consolidate the entangled-enhancement definition** into System model.
- Keep the honest framing already added (δ₁<0 ⇒ not strict entangled
  enhancement; open-loop-L2 caveat; ρ not cleanly measurable in the single run).
  The §3 campaign should *supply* the case where δ₁>0 so a **clean** entangled
  enhancement can finally be shown — this is the strongest possible answer to
  both Q1 and the "real entangled enhancement" core-story point.
- Page budget: both structures likely exceed 8 pages with new results. Decide
  the venue/limit early (the current file is tuned to 8pp for a conference).

---

## 6. Writing-improvement plan (W1, W2)

These are line-level edits, applied after the structure is settled (so we don't
polish text that gets moved/cut).

### 6.1 W1 — de-chain overloaded sentences (`:` / `;` abuse)
- Do a pass flagging every `;` and mid-sentence `:` that joins *distinct
  topics*; split each into 2–3 sentences with a connective and a little
  elaboration.
- Concrete first target (sensei's example), rewrite e.g.:
  > *Current:* "The focus is on C1, the retrained unit, and C2, its direct
  > consumer, whose input interface carries the cascade; C3 gives the
  > system-level reading."
  > *Revised:* "The experiments focus on two components. C1 is the retrained
  > unit, and C2 is its direct consumer. The interface between them, the stream
  > of detected objects, is where the cascade travels. C3 is kept only as the
  > system-level readout that tells us whether the change helps or harms the
  > final plan."
- Sweep the whole paper; the Intro, Methodology, and Results have the densest
  offenders.

### 6.2 W2 — add introductory context before terse claims
For each flagged terse sentence, add a lead-in that says *what it is* and *why it
matters* before the claim. Targets and intended fix:
- "This paper supplies that measurement." → precede with one sentence naming
  *what* measurement (the missing controlled AV-pipeline experiment) so the
  pronoun "that" is grounded.
- "The main instrument is dual-mode evaluation:" → first say *why* we need two
  modes (to separate a component's own error from inherited upstream error),
  then name the instrument.
- "The gap between the modes isolates the cascade contribution…" → define the
  gap in words first (isolated = own error; pipeline = own + inherited; the
  difference = inherited), then give the formula.
- "The harness computes the five-quantity profile of Section III-D…" → remind
  the reader what the five quantities are and why (one C1 metric + two C2 modes
  + two C3 modes) before referencing the section.
- "The three isolated-mode rows are exactly zero-delta." → lead with what the
  reader should look for (the frozen components must be unchanged) and why it is
  a *validity check*, not a result.
- "Table I exhibits a second inversion." → name the *first* inversion explicitly
  and say "inversion of what," then present the second.

### 6.3 Style guardrails to keep
- Keep the earlier fixes intact: no em dashes in prose, no marketing overclaims,
  no "free-tier/blog" self-promotion, single-block author, YOLOv11n naming.

---

## 7. Proposed execution order (phased)

> Each phase gated on the previous; compute-heavy work only after decisions.

**Phase 0 — Decisions (no compute).**
- Confirm A-vs-B strategy ("target A, fall back to B") with sensei.
- Confirm the campaign matrix scope (§3.2) and the target venue/page limit.

**Phase 1 — Q1 experiments (compute).**
- Add a config-loop driver over the existing harness; run the campaign matrix
  (§3.2), ≥3 seeds per config, log δ₁/Δ₂/Δ₃/ρ/collision + per-scene.
- Produce the regime table + (δ₁, Δ₃)/ρ figure.
- **Milestone:** at least one config with δ₁>0 (a *clean* entangled enhancement)
  — the key asset for the core story.

**Phase 2 — Q2 CARA evaluation (compute + code).**
- Implement `drift_gate.py`; evaluate retrospectively on the Phase-1 runs
  (§4.1), threshold sweep.
- Evaluate the full admission rule over the Phase-1 updates (§4.2).
- **Milestone:** decide A vs. B based on how convincing Phase 2 is.

**Phase 3 — Restructure the paper (writing).**
- Apply the chosen structure (§5.2 or §5.3): move Related Work to back,
  consolidate the definition into System model, (A) promote CARA, or (B) add
  Practical implications.
- Insert Phase-1/Phase-2 results.

**Phase 4 — Writing pass (W1, W2).**
- De-chain overloaded sentences; add context to terse ones; keep style
  guardrails. Recompile, re-check page budget, references, figures.

**Phase 5 — Review & submit.**
- Full read-through against the 3 core-story points; sensei review; finalize.

---

## 8. Open questions to raise with sensei

1. **A or B?** Confirm "target A, fall back to B," or an explicit choice.
2. **Venue and page limit?** (Drives how much of the campaign matrix fits.)
3. **Compute ceiling?** Still Kaggle free tier only, or any HPC now available?
   (Affects whether Singapore↔Boston reverse-shift and larger seed counts are in
   scope, and whether we can approach a NAVSIM-grade terminal metric for CARA.)
4. **Is n=3 validation acceptable for the CARA evaluation**, framed as proof of
   concept, or does the CARA evaluation need a larger split to be credible?

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| No config yields δ₁>0 (clean entangled enhancement) | Widen the matrix (class-balanced fine-tune, reverse shift); if still none, report the *conditions under which* it does/does not appear — still a Q1 answer |
| CARA evaluation weak on n=3 | Pre-commit to fallback structure (B); frame CARA as proof-of-concept |
| Campaign exceeds 8-page conference limit | Decide venue early; move detailed tables to an appendix or a longer-format venue |
| Seed variance swamps the regime signal | Report mean ± std; treat "fragile to seed" as a finding, not a failure |

---

*Plan prepared for review. No experiments or paper edits executed yet.*
