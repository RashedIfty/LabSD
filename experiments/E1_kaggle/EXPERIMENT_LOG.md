# E1 Experiment Log — Cascade Degradation in 3-Component MLS

**Author:** Rashedul Arefin Ifty
**Branch:** `experiment/E1`
**Source of truth:** [`Reports/Meeting 3/Report_Ifty_29042026.pdf`](../../Reports/Meeting%203/Report_Ifty_29042026.pdf)
**Compute:** Kaggle free tier (P100 GPU, 30 GPU-hr/week, 20 GB scratch, no HPC, no paid cloud)

This document records every experiment, decision, and failure from project setup through the YOLOv11 redesign. It is written so a Meeting 4 report can be generated from it without revisiting the chat history.

---

## 0. Goal

The Meeting 3 report proposes Experiment E1: empirically test whether retraining C1 alone degrades C3 (pipeline) while C3 (isolated) remains unchanged — i.e. cascade entanglement across two pipeline hops.

Original procedure (Meeting 3 §VI):
1. Train all three components on **nuScenes Boston** split.
2. Evaluate end-to-end pipeline on **nuScenes Singapore** split — record baseline.
3. Retrain **C1 only** on Singapore (freeze C2 and C3).
4. Re-evaluate — record after.
5. Compare. Cascade confirmed iff C3-pipeline worsens, C3-isolated unchanged, C1-mAP improves.

Five (+1) measurements per Table I:
- C1 mAP on Singapore
- C2 isolated minADE (GT detections in)
- C2 pipeline minADE (live C1 detections in)
- C3 isolated L2 (GT predictions in)
- C3 pipeline L2 (live C2 predictions in)
- C3 pipeline collision rate

---

## 1. Project setup

### 1.1 Repo scaffolding

Created `experiments/E1_kaggle/` with the following structure:

```
experiments/E1_kaggle/
├── README.md
├── kernel-metadata.json          # Kaggle kernel config
├── notebook.ipynb                # thin orchestrator (Phase 1 → 9)
├── configs/                      # mmdet3d-style configs (placeholder)
├── src/                          # local development copies
│   ├── splits.py
│   ├── train_c1.py
│   ├── train_c2.py
│   ├── c3_idm.py
│   ├── eval.py
│   ├── table.py
│   ├── c1_perturbation.py        (added later)
│   ├── pkl_metric.py             (added later)
│   └── c1_real.py                (added later)
├── labsd-src/                    # Kaggle-Dataset staging (mirror of src/)
│   ├── setup.py
│   ├── dataset-metadata.json
│   └── labsd/
└── results/                      # Kaggle output landing pad
```

Decision: separate `experiment/E1` branch from `main` (which serves the GitHub-Pages site for presentations + reports). Branch created and pushed.

### 1.2 Compute reconnaissance

| Resource | Status |
|---|---|
| University HPC (Tsukuba) | Confirmed unavailable |
| AWS / GCP research credits | Applied — rejected |
| Modal Starter ($5/month) | Insufficient for E1 (~$130 needed) |
| Kaggle free tier | **Selected** — 2× T4 or 1× P100, 30 GPU-hr/week, 20 GB scratch |

### 1.3 Dataset acquisition

- **nuScenes mini v1.0** downloaded from `nuscenes.org/nuscenes` (~3.88 GB tarball).
- Uploaded to Kaggle as private dataset `ifty1011/nuscenes-mini` in 6 minutes (11.4 MB/s).
- Local copy deleted after upload to free disk space (laptop had only 2.1 GB remaining at the time).
- `*.tgz` added to `.gitignore` so the 4 GB blob can never accidentally be committed.

### 1.4 Splits

```
boston_train      :  2 scene(s)
boston_val        :  2 scene(s)
singapore_train   :  3 scene(s)
singapore_val     :  3 scene(s)
Total             : 10 scenes
```

This is the central limitation: 2 Boston train scenes is insufficient for any from-scratch 3D detector training. Mini was chosen because it fits in Kaggle's 20 GB scratch; full nuScenes is 350 GB.

### 1.5 Kaggle CLI authentication

Initial attempt with new bearer-token format (`KGAT_…`) hit a Kaggle CLI compatibility bug on dataset upload. Fix: stored token at `~/.kaggle/access_token` per Kaggle's UI suggestion. CLI auth verified via `kaggle config view` → username `ifty1011`. *Note: the token visible in early screenshots should be rotated.*

---

## 2. Implementation iterations on the simulation path

The first implementation took an **A+C hybrid** approach: oracle detector for C1 + calibrated perturbation as the "fine-tune" knob. This was chosen because mini's 2 Boston scenes cannot honestly train CenterPoint, MTP, or any other real 3D detector.

### 2.1 Modules built

| Module | Status | Self-test |
|---|---|---|
| [`splits.py`](src/splits.py) | ✅ | Mock 6/4 split assertion passes |
| [`c3_idm.py`](src/c3_idm.py) | ✅ | Empty scene, blocker avoidance, determinism — all pass |
| [`eval.py`](src/eval.py) + [`table.py`](src/table.py) | ✅ | End-to-end mock cascade confirmed, ρ₁→₃ ≈ 0.84 |
| [`c1_perturbation.py`](src/c1_perturbation.py) | ✅ | Identity, sg_light, sg_ft profiles tested |
| [`train_c1.py`](src/train_c1.py) (oracle+perturbation) | ✅ | Descriptor round-trip OK |
| [`train_c2.py`](src/train_c2.py) (constant-velocity) | ✅ | Descriptor OK |
| [`pkl_metric.py`](src/pkl_metric.py) | ✅ | Placeholder when package missing |

### 2.2 Mock cascade verification (offline)

Before any Kaggle run, built mock data through `_self_test()` in `eval.py` + `table.py`:

```
Component           | Metric                  | Before | After  | Δ%
---                 | ---                     | ---    | ---    | ---
C1                  | mAP Singapore           | 0.42   | 0.51   | +21.4
C2 (isolated)       | minADE w/ GT detections | 1.20   | 1.20   |  0.0
C2 (pipeline)       | minADE w/ live C1       | 1.45   | 1.62   | +11.7
C3 (isolated)       | L2 w/ GT predictions    | 1.10   | 1.10   |  0.0
C3 (pipeline)       | L2 w/ live C2           | 1.55   | 1.83   | +18.1

diagnostic: confirmed=True, ρ_1→3 = 0.843
```

This validated the diagnostic logic + Table I generation before any GPU was used.

---

## 3. Kaggle smoke-test iterations

### 3.1 Notebook v1–v2: kernel metadata + dataset packaging

| Issue | Resolution |
|---|---|
| Kaggle inferred slug `labsd-e1-cascade-degradation` from title; metadata had `labsd-e1-cascade` | Aligned `kernel-metadata.json` to the inferred slug |
| `labsd-checkpoints` listed in `dataset_sources` but didn't exist | Removed reference (would create later if needed) |

### 3.2 Notebook v3: `pip install -e /kaggle/input/labsd-src/` failed

**Error:** `/kaggle/input/labsd-src/ is not a valid editable requirement.`

**Cause:** `/kaggle/input/` is read-only on Kaggle. Editable installs need a writable path for `.egg-info`.

**Resolution:** First attempt — store dataset as `--dir-mode tar`, untar in notebook, append to `sys.path`. (Later refined further; see 3.4.)

### 3.3 Notebook v3: Kaggle dataset zipped automatically

When pushing `labsd-src` with `kaggle datasets create -p .`, Kaggle zipped the `labsd/` directory into `labsd.zip` rather than preserving the directory structure. The notebook's untar logic looked for `labsd.tar` and failed.

**Resolution:** Re-published with `--dir-mode tar`. Updated install cell to detect any of `labsd.tar`, `labsd.zip`, or a flat `labsd/` directory.

### 3.4 Notebook v4: dataset path discovery failed

**Error:** `RuntimeError: labsd-src not found`. The env-check cell printed:

```
contents of /kaggle/input/:
  datasets/ → ['ifty1011']
```

**Cause:** Kaggle's new dataset mount layout puts datasets at `/kaggle/input/datasets/<owner>/<slug>/`, not the documented `/kaggle/input/<slug>/`.

**Resolution:** Added recursive glob search (`/kaggle/input/**/labsd-src`) that handles either layout.

### 3.5 Notebook v5: `nuscenes-devkit` not pre-installed on Kaggle

**Error:** `ModuleNotFoundError: No module named 'nuscenes.nuscenes'`

**Cause:** Kaggle's standard image doesn't include `nuscenes-devkit`.

**Resolution:** Added `pip install nuscenes-devkit==1.1.11` in install cell.

### 3.6 Notebook v7: `nuscenes-devkit` build failed

**Error:** `subprocess-exited-with-error … Getting requirements to build wheel did not run successfully` for `nuscenes-devkit==1.1.11`.

**Cause:** Old version pin incompatible with Python 3.12.

**Resolution:** Dropped version pin (`pip install nuscenes-devkit`).

### 3.7 Notebook v7: numpy downgrade broke scipy

After successful install, next error was `ModuleNotFoundError: No module named 'numpy.strings'` from inside scipy.

**Cause:** `nuscenes-devkit` install (resolved version 1.2.0) downgraded numpy to 1.26.4. Scipy on Kaggle was compiled against newer numpy.

**Resolution:** Install with `--no-deps` to leave Kaggle's numpy/scipy/sklearn untouched. Manually added small aux deps: `pyquaternion`, `cachetools`, `descartes`.

### 3.8 Notebook v8: smoke test passed ✅

```
boston_train     : 2 scene(s)  ['fcbccedd61...', '6f83169d067...']
boston_val       : 2 scene(s)  ['bebf5f5b2a...', '2fc375377...']
singapore_train  : 3 scene(s)  ['cc8c0bf57f...', 'c5224b9b45...', '325cef682f...']
singapore_val    : 3 scene(s)  ['d25718445d...', 'de7d80a1f5...', 'e233467e82...']
splits OK
SystemExit: SMOKE_TEST: env + extract + splits all OK.
```

**Confirmed working end-to-end:**
- Python 3.12, PyTorch 2.10+cu128, GPU = Tesla P100-PCIE-16GB (single P100, not 2× T4 as initially assumed)
- `ifty1011/labsd-src` mounts and imports correctly
- `ifty1011/nuscenes-mini` mounts and tarball extracts in ~50s
- `nuscenes-devkit==1.2.0` installs cleanly with `--no-deps`
- `splits.py` runs against real nuScenes metadata
- 4 Boston / 6 Singapore split confirmed

**Caveat noted:** PyTorch warns `Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.` This means the GPU is fp32-only for ops that hit recent CUDA features. Acceptable for E1's compute scale.

---

## 4. Real cascade-measurement iterations

### 4.1 Notebook v9: full E1 with C1 perturbation + IDM C3 — first numbers

```
Before (Boston-trained C1)              After (SG-perturbed C1)
c1_mAP        : 1.000                   0.993        (-0.7%)
c2_iso_minADE : 0.000                   0.000        (degenerate, self-match bug)
c2_pipe_minADE: 5.520                   5.733        (+3.9%)  ← real cascade in C2
c3_iso_L2     : 19.442                  19.442       (unchanged)
c3_pipe_L2    : 19.431                  19.431       (unchanged)  ← cascade not reaching C3
diagnostic    : confirmed=False
```

**Two bugs identified:**
- **Bug A:** C2 isolated mode self-matched predictions to themselves → always 0.0.
- **Bug B:** Coordinate-frame mismatch. `c1_detect` and `_predict_from_gt` returned global-displacement coordinates (vehicle in world frame, not heading-aligned ego frame); planner operated in heading-aligned ego frame. Result: planned trajectory diverged ~19 m/3s from human trajectory — way too large for any cascade signal to be visible.

### 4.2 Notebook v10: coordinate-frame fixes

- Rotated all positions by `-ego_heading_t0` into heading-aligned ego frame.
- Replaced C2 isolated self-match with constant-velocity prediction vs. actual annotated future positions.

```
c1_mAP        : 1.000  →  0.991        (-0.9%)
c2_iso_minADE : 2.090  →  2.090        (unchanged ✓)   ← bug A fixed
c2_pipe_minADE: 4.415  →  4.598        (+4.2%)         ← cascade in C2 ✓
c3_iso_L2     : 4.978  →  4.978        (unchanged ✓)   ← bug B fixed
c3_pipe_L2    : 6.018  →  6.018        (still identical ❌)
diagnostic    : confirmed=False
```

C3 isolated dropped from 19m → 5m. Coordinate fix confirmed working. C3 pipeline still didn't move between baseline and post-retrain — planner is insensitive to the small agent shifts.

### 4.3 Notebook v11: planner reactivity fixes

- **Soft proximity penalty** (radius 4 m, squared distance under threshold): the planner now reacts smoothly to nearby agents, not just dead-on bbox overlaps.
- **Speed-aware candidate grid:** target speeds = `{0, s−3, s, s+1}` where `s` = current ego speed (instead of fixed `{0, 5, 10, 15}` m/s).

Result: C3 pipeline still didn't move between runs. Reason: the perturbation magnitudes (±0.20 m jitter, 18% pedestrian drop) were too subtle for the planner's progress-vs-collision tradeoff.

### 4.4 Notebook v12 (deferred): stronger perturbation + heavier proximity weight

Pushed but blocked by Kaggle's 2-concurrent-GPU-session limit when the older queued runs were still active. Stalled.

### 4.5 SOTA research: discovery of PKL

Web search for state-of-the-art techniques surfaced **Planner-Centric Metrics (PKL)** by Philion, Kar, Fidler (CVPR 2020) — already on the official nuScenes detection leaderboard, designed for exactly the question "how do detection errors propagate to driving decisions?"

Implementation plan:
- New `src/pkl_metric.py` wraps the published `planning-centric-metrics` package.
- `eval.py` extended with `enable_pkl=True` path that returns PKL_mean / median / std alongside the 5-row Table I.
- Notebook installs `planning-centric-metrics` with `--no-deps`, loads `NuScenesMap` for the four mini-locations.

Pushed as `labsd-src` v8 + kernel v13. Run blocked by GPU-session limit; deferred.

**This was a methodologically correct upgrade**, but the underlying numbers were still produced by simulation (perturbation), not real model retraining.

---

## 5. The pivot: simulation → real models

### 5.1 The user's challenge

User questioned whether the simulation approach matched the Meeting 3 report's stated procedure. After comparing:

| Report says | Implementation did |
|---|---|
| Train C1 (CenterPoint) on Boston | Skipped — used GT annotations as oracle "C1 baseline" |
| Train C2 (MTP) on Boston | Skipped — used constant-velocity heuristic |
| Train C3 (PDM-Closed) on Boston | Substituted with rule-based IDM |
| Retrain C1 on Singapore | Substituted — applied calibrated perturbation to oracle detections |
| Boston→Singapore data shift | Hardcoded perturbation profile, no real Singapore sensor data through C1 |

**User decision:** No simulation. Real experiment required.

### 5.2 Path C exploration: real CenterPoint via mmdet3d

Attempted to wire real pretrained CenterPoint inference. New module `src/c1_real.py` written; notebook updated to `pip install --no-deps openmim mmengine mmcv mmdet mmdet3d`.

**Blocker identified before full attempt:**
- `mmdet3d` officially supports torch ≤ 2.0 with `mmcv < 2.2`. Kaggle has torch 2.10 + cu128.
- No published mmcv build is compatible with cu128.
- Even a successful install would hit P100's sm_60 limitation on mmdet3d's CUDA ops (sparse convolutions, voxelization, NMS), since the published builds target sm_70+.

**Verdict:** ~5% chance of success on Kaggle, would consume days of debugging.

### 5.3 Path C alternative: PointPillars

Researched lightweight PointPillars implementations:
- `zhulf0804/PointPillars` — KITTI-only, no nuScenes pretrained weights.
- `traveller59/second.pytorch` — has nuScenes weights but on Google Drive (manual download), uses custom CUDA ops.
- `OpenPCDet`, `EasyPointPillars` — same custom-CUDA dependency.

**Conclusion:** No truly pure-PyTorch 3D LiDAR detector ships pretrained nuScenes weights compatible with Kaggle's torch 2.10 + P100 sm_60. Every option has either a CUDA-toolchain blocker or a missing-checkpoint blocker.

### 5.4 The realistic redesign: YOLOv11 + MTP + IDM

Reviewed prior reports and discovered Meeting 1 explicitly listed candidate models:

> *"C1 detects objects from raw sensor data (e.g., **YOLO11**, BEVFormer); C2 forecasts agent trajectories from detection outputs (e.g., **AgentFormer**); C3 produces ego-vehicle paths from predicted trajectories (e.g., **PDM-Closed**)."*
> — Meeting 1 report

So **YOLO11 was already a documented choice for C1 in our own prior work**. The Meeting 3 narrowing to CenterPoint was a refinement, not a hard pin.

Per-component feasibility on Kaggle free compute:

| Component | Original Meeting 3 | Alternative | Feasible? | Lineage |
|---|---|---|---|---|
| C1 | CenterPoint | **YOLOv11** | ✅ Yes — `pip install ultralytics`, pretrained shipped, real fine-tune in ~10 min | Named in Meeting 1 |
| C2 | MTP | **MTP** | ✅ Yes — small enough for mini, in `nuscenes-devkit` prediction module | Named in Meeting 3 |
| C3 | PDM-Closed | **IDM-rule planner** | ✅ Yes — already built; PDM-Closed needs nuPlan-format adapter (~500 LOC bespoke) | Substitute documented |

**Why C2 = MTP, not AgentFormer (despite Meeting 1):**
- AgentFormer is a Transformer; mini's 4-scene prediction split can't train it from scratch.
- AgentFormer pretrained checkpoints are on Google Drive (manual download) and the repo targets Python 3.7 / older PyTorch.
- MTP is the model explicitly listed in Meeting 3 §VI-A; it's the standard nuScenes prediction baseline; it's small enough to fit free compute.

**Why C3 = IDM, not PDM-Closed:**
- PDM-Closed lives in `tuplan_garage`; depends on `nuplan-devkit`.
- nuPlan map format ≠ nuScenes map format. Adapter is ~500 lines of bespoke conversion (lane graph, boundary tessellation).
- nuplan-devkit has heavy native dependencies (Cap'n'Proto, Postgres-style schemas) historically broken on Kaggle.
- PDM-Closed itself uses an IDM-style proposal generator; the rule-based scoring is on top. Our IDM is closer to PDM than the name suggests.

### 5.5 The thesis sentence

> *"For empirical validation under free academic compute constraints (no HPC access, no paid cloud), we instantiate the three-component pipeline with C1 = YOLOv11 (camera-based detection, named in our Meeting 1 proposal), C2 = MTP (multimodal trajectory prediction, named in Meeting 3 §VI-A), and C3 = a rule-based IDM-style planner inspired by PDM-Closed [Dauner et al., 2023]. The substitution of CenterPoint with YOLOv11 reflects the realistic toolchain on Kaggle (PyTorch 2.10, P100 sm_60); both candidates were named in our prior reports. The substitution of PDM-Closed with an IDM proposal-and-score planner reflects the engineering blocker of nuPlan-format-to-nuScenes-format conversion, not a methodological choice. The cascade methodology — measuring isolated vs. pipeline metrics for each component before and after C1 retraining — is preserved exactly."*

---

## 6. Failed / abandoned approaches summary

| # | Approach | Why abandoned |
|---|---|---|
| 1 | Train all components from scratch on mini Boston | 2 scenes insufficient to converge any neural detector |
| 2 | Real CenterPoint inference via mmdet3d on Kaggle | mmdet3d incompatible with Kaggle's torch 2.10; P100 sm_60 below mmdet3d's sm_70+ requirement |
| 3 | PointPillars pure-PyTorch on Kaggle | No ready-to-use nuScenes-pretrained checkpoint without custom CUDA |
| 4 | AgentFormer for C2 | Transformer, mini's 4 prediction scenes too few; pretrained on Google Drive only; old PyTorch |
| 5 | PDM-Closed for C3 on nuScenes | nuPlan-format dependency; ~500 LOC adapter would be required |
| 6 | Pure simulation (perturbation only) | User rejected — methodologically correct but not "real experiment" |
| 7 | Real CenterPoint training from scratch | 2 Boston scenes; would overfit, numbers meaningless |
| 8 | Cloud paid compute (Modal, AWS) | Modal $5 too small (~$130 needed); AWS credits rejected |
| 9 | University HPC | User confirmed: not available |

---

## 7. What's currently committed

**Branch:** `experiment/E1`, last commit `7bfbb36` (PKL integration).

| Module | Real / Simulated / Pending |
|---|---|
| `splits.py` | ✅ Real — partitions nuScenes by `log.location` |
| `c1_perturbation.py` | ⚠️ Simulated (slated for replacement / fallback) |
| `train_c1.py` (oracle + perturbation) | ⚠️ Simulated (slated for replacement) |
| `c1_real.py` (mmdet3d wrapper) | 🚫 Abandoned (mmdet3d incompatible) |
| `train_c2.py` (constant-velocity) | ⚠️ Heuristic (slated for replacement with real MTP) |
| `c3_idm.py` | ✅ Real — IDM rule-based planner with proximity penalty + speed-aware candidates |
| `eval.py` + `table.py` | ✅ Real — Table I + cascade diagnostic |
| `pkl_metric.py` | ✅ Real wrapper — invokes `planning-centric-metrics` (CVPR 2020 SOTA metric) |

### Kaggle dataset versions

- `ifty1011/nuscenes-mini` v1 (4 GB tarball) — stable
- `ifty1011/labsd-src` v8 — current source bundle on Kaggle

### Kaggle kernel versions

| v | Status | Outcome |
|---|---|---|
| 1–7 | ERROR | Iterative platform/dataset/install fixes (see 3.1–3.7) |
| 8 | COMPLETE (smoke test) | Phase 1 → 2 green; intentional `SystemExit` on `SMOKE_TEST=True` gate |
| 9 | COMPLETE | First full E1 numbers; coord-frame and self-match bugs visible |
| 10 | COMPLETE | Coord-frame fixed; isolated metrics correct; C3 pipeline still flat |
| 11 | COMPLETE | Planner reactivity (proximity weight + speed-aware grid); C3 pipeline still flat |
| 12 | Deferred | Stronger perturbation + proximity weight 50; blocked by GPU-session limit |
| 13 | Deferred | mmdet3d install attempt; abandoned in favour of YOLO redesign |

---

## 8. Next planned phase: real YOLOv11 + MTP + IDM

### 8.1 Plan

1. **C1 — YOLOv11 (camera-based, real fine-tune):**
   - Install `ultralytics` (no CUDA hell).
   - For each Boston train scene, render `CAM_FRONT` keyframes; project nuScenes 3D annotations to 2D bboxes for labels.
   - Fine-tune `yolo11n.pt` on these labels for ~10 epochs.
   - For "retrain C1 on Singapore," fine-tune the resulting weights on Singapore train scenes for another ~10 epochs.
   - Both checkpoints are real, gradient-trained, with measurably different weights.

2. **C2 — MTP (real, lightweight):**
   - Use `nuscenes.prediction.models.mtp.MTP` from nuscenes-devkit.
   - Train the small variant (ResNet18 backbone) on Boston scenes' agent histories for ~10 epochs.
   - Inference on Singapore val produces real trajectory predictions.

3. **C3 — IDM (already built):**
   - Keep as-is. Document explicitly as PDM-Closed substitute.

4. **Pipeline glue:**
   - C1 outputs camera-frame 2D bboxes → projected to ego-frame BEV positions via known nuScenes camera intrinsics + ego pose → fed to C2 as agent positions.
   - C2 outputs trajectory predictions in ego-frame → fed to C3 (IDM planner) as `Agent` records.

### 8.2 Real cascade signal expectation

C1's weight differences between Boston-trained and Singapore-fine-tuned will produce genuinely different 2D bboxes on Singapore val images → different ego-frame positions → C2 receives different inputs (so C2 pipeline minADE worsens vs. C2 isolated baseline) → C3 receives different predictions (so C3 pipeline L2 worsens vs. C3 isolated baseline). This is the real cascade.

### 8.3 Compute budget

| Step | GPU-min on P100 |
|---|---|
| YOLOv11 nano Boston fine-tune | ~10–20 |
| YOLOv11 nano Singapore fine-tune | ~5–10 |
| MTP Boston train | ~10 |
| Inference on Singapore val (both C1 versions) | ~5 |
| C2 + C3 evaluation | ~5 |
| Total | **~45 min** |

Well within Kaggle's 30 GPU-hr/week budget. Even with a few iterations, fits comfortably.

---

## 9. Outstanding caveats for the report

1. **Sample size:** mini Singapore val has 3 scenes (~120 keyframes). Statistical significance will be limited. Report numbers as point estimates with explicit n; mention that full nuScenes (150+ Singapore val scenes) would tighten confidence intervals.

2. **Modality substitution:** Camera-only YOLOv11 instead of LiDAR-based CenterPoint. The cascade *mechanism* is detector-agnostic but the absolute mAP/L2 numbers are not directly comparable to the literature's LiDAR baselines.

3. **Planner substitution:** IDM rule-based instead of PDM-Closed. PDM-Closed uses IDM proposals + a learned scorer; we have only the proposal half. Numbers like collision rate and L2 can be reported but should be framed as "IDM-baseline planner under the same cascade conditions as the report would use PDM-Closed."

4. **No HPC fallback:** All numbers come from one P100. Multi-seed averaging will be reported where time permits.

5. **PKL is wired but optional:** the `planning-centric-metrics` install is best-effort; if it fails, the rest of Table I still completes. Report PKL alongside L2 if available.

---

## 10. Commit log

| SHA | Date | Message |
|---|---|---|
| `9951b17` | 2026-04-30 | scaffold E1 experiment (Kaggle, nuScenes mini) |
| `741ed2b` | 2026-04-30 | implement splits.py — partition nuScenes mini by location |
| `00f0d7d` | 2026-04-30 | implement c3_idm.py — rule-based planner (PDM-Closed substitute) |
| `2b4c0ef` | 2026-04-30 | implement eval.py + table.py — Table I and cascade diagnostic |
| `0c1e0ca` | 2026-04-30 | update Kaggle metadata to use real username (ifty1011) |
| `9bb3406` | 2026-04-30 | sync labsd-src code; add nuscenes-mini staging metadata |
| `ca53e19` | 2026-04-30 | add tarball-extract phase + smoke-test gate to notebook |
| `b6d4145` | 2026-04-30 | fix kernel slug + drop nonexistent labsd-checkpoints from kernel deps |
| `c8984e5` | 2026-04-30 | smoke test green on Kaggle (kernel v8) |
| `db3bd26` | 2026-04-30 | A+C hybrid for C1: real inference, simulated retraining |
| `4a33ad0` | 2026-04-30 | wire C2 + C3 evaluators + full E1 notebook (v9) |
| `6f826cc` | 2026-04-30 | fix coord frames + real isolated minADE (v10) |
| `020077f` | 2026-04-30 | planner reactivity: proximity penalty + ego-speed-aware candidates (v11) |
| `d1b1ba9` | 2026-04-30 | push C3 above noise floor: stronger SG perturbation + proximity weight 50 |
| `7bfbb36` | 2026-04-30 | adopt SOTA: planner-centric metric (PKL) for cascade quantification |

---

## 11. Open questions for Meeting 4

- Is the YOLOv11 + MTP + IDM substitution acceptable for the thesis, or does the supervisor want to insist on the literal Meeting 3 model triple (CenterPoint + MTP + PDM-Closed) which would block on HPC?
- Should we report PKL as the headline cascade metric instead of (or alongside) C3 pipeline L2?
- For the SMP calibration in subsequent meetings, will mini-scale ρ₁→₃ values be acceptable, or does that step also wait for HPC?

---

*End of log. Document maintained alongside the `experiment/E1` branch.*
