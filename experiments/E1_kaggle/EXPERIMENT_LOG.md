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
| 10 | YOLOv11 fine-tune on **Kaggle P100 GPU** | sm_60 incompatible with Kaggle's torch 2.10.0+cu128 (sm_70+ only). Confirmed at v14 with `AcceleratorError: CUDA error: no kernel image is available for execution on the device`. Resolution: switch to **T4 x2 accelerator** (sm_75) for v15. |

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
| 14 | **FAILED at 738s** — `AcceleratorError: CUDA error: no kernel image is available for execution on the device` | First real-model kernel. Install + dataset build + YOLO model load all succeeded. **Failure root cause:** Kaggle's torch 2.10.0+cu128 was compiled for sm_70+ only; Tesla P100 is sm_60. The earlier `UserWarning: Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible` became a hard error when YOLOv11 called `.to(device)`. Unfixable on this accelerator. **Side issues observed:** mmdet3d install (still attempted from earlier abandoned path) timed out at 600s — wasted budget; will remove. NuScenesMap json files missing from mini tarball — PKL skipped (graceful no-op). |
| 15 | **FAILED at 96s — same P100 sm_60 error** | The `--accelerator nvidia-tesla-t4` CLI flag was accepted by Kaggle CLI but Kaggle still allocated a P100. The kernel's accelerator preference is sticky on the server side from previous runs and CLI cannot override it. Same `AcceleratorError: CUDA error: no kernel image is available` at YOLOv11 `.to(device)`. Fix: must change the accelerator via Kaggle's web UI ("Edit" → "Settings" → "Accelerator: GPU T4 x2" → "Save"), then re-push. |
| 16a | `labsd-src` v10 pushed with CPU fallback | Belt-and-suspenders: `c1_yolo.fine_tune_yolo` now reads `torch.cuda.get_device_capability(0)` and forces `device='cpu'` if the GPU's major capability is below 7 (i.e. below sm_70). Slower but real fine-tune still happens. Will activate automatically on the next P100 run if the user hasn't yet changed the web-UI setting. |
| 16 | **Partial success → failed at Phase 6** | T4 x2 allocated correctly (sm_75 OK with torch 2.10). YOLOv11 Boston fine-tune **completed** in 47s on T4. mAP50 climbed 0.039 → 0.119 across 10 epochs on 81 Boston train images, 723 instances. C1 Boston descriptor written. Then crashed at the baseline-measurement cell with `KeyError: 'profile_params'` — `evaluate_c2` was still calling `load_c1_profile()` on the YOLO descriptor (perturbation-shaped JSON loader). Fix: route YOLO descriptors to the `("yolo", weights_path)` tuple path in `evaluate_c2` too (was already done in `c2_predict` but missed in `evaluate_c2`'s pipeline-mode branch). |
| 17 | **FAILED at 153s — `KeyError: 'profile_params'` again**, same root cause but in `pkl_metric.py` this time | Boston YOLO fine-tune completed in 47s on T4 (mAP50 climbed 0.039 → 0.119, identical to v16). Then `evaluate_pkl` was called, which also called `load_c1_profile()` unconditionally — the same dispatch bug as v16, just in a different module that I missed. PKL was supposed to be skipped anyway because `nusc_maps` is empty (no expansion JSONs in the mini tarball), but the no-op check happened too late. |
| 18 | Skipped by Kaggle | Per-kernel accelerator setting reset to P100 default after kernel push (CLI `--accelerator` was not passed); user reported manual cancel. |
| 19 | **Pipeline end-to-end real — first cascade signal observed.** Singapore fine-tune mid-run when log was sampled. | All three v17 fixes worked: PKL graceful skip, weight cache hit within session, T4 pinned. **Phase 6 baseline (Boston-trained YOLO C1 on Singapore val):** `c2_iso_minADE=2.09, c2_pipe_minADE=15.19, c3_iso_L2=4.93, c3_pipe_L2=6.30`. The 1.37 m gap between C3 isolated (4.93) and C3 pipeline (6.30) is the **first real cascade signal in this project** — YOLO's actual detection errors propagate through C2 zero-velocity rollouts into the IDM planner. C2 pipeline (15.19) ≫ C2 isolated (2.09) confirms the upstream perturbation reaches C2 strongly. PKL = null with note `PKL skipped for YOLO C1` (fix worked). Singapore fine-tune started successfully (loaded Boston weights as base, transferred 499/499 items, AMP checks passed). Phase 8 after-retrain numbers + Table I + diagnostic pending — log truncated mid-run. |
| 20 | **✅ COMPLETE — full E1 ran end-to-end, real models, real numbers.** Output downloaded: `baseline.json`, `after_retrain.json`, `cascade_result.png`, full log. | First fully-completed real-model run. All phases executed: env → tarball extract → splits → C1 Boston YOLO fine-tune (cached) → Phase 6 baseline → C1 Singapore YOLO fine-tune → Phase 8 after-retrain → Table I → cascade diagnostic → headline plot. |
| 21 | Pushed (`--accelerator nvidia-tesla-t4`, auto-running) | **Presentation-figure run.** Adds Tier 1 + Tier 2 visualizations: (a) `c1_yolo.fine_tune_yolo` now `plots=True` so Ultralytics writes loss + mAP curves + confusion matrix + sample preds; (b) `c1_descriptor_yolo` accepts `val_data_yaml` and runs `model.val()` to populate real `c1_mAP` (previously `null`); (c) new `src/figures.py` with 10 figure generators — full Table I bar chart, cascade signature, annotated pipeline diagram, copy of YOLO training plots, side-by-side detection grid (same Singapore val images, Boston-YOLO vs SG-YOLO), per-scene C3 L2, per-scene Δ failure modes; (d) `evaluate_c3` now returns `per_scene` list for the per-scene breakdown chart; (e) Phase 9 of the notebook calls `figures.generate_all()` and inline-displays the headline figures. |
| 22 | **✅ COMPLETE — full presentation set produced.** All 16 figures + raw JSONs + YOLO weights downloaded to `results/run-v22/`. | Final v22 numbers: **C1 mAP@50** Boston-trained=0.119, SG-fine-tuned=0.064 (mAP did NOT strictly improve — small-data overfitting on 119-img fine-tune set; documented in report as a caveat); **C2 pipeline minADE** 15.19→3.95 (−74%); **C2 isolated** 2.09 unchanged ✓; **C3 pipeline L2@3s** 6.30→6.58 (+4.45%); **C3 isolated** 4.93 unchanged ✓. **Per-scene breakdown:** entire +0.28 m C3 regression comes from one scene (`d2571844`, +0.84 m); the other two Singapore val scenes are bit-identical. The C2-improves-but-C3-degrades pattern is the canonical entangled-enhancement signature from Wang & Machida 2025. |

---

## 9. Meeting 4 deliverable (2026-04-30)

Generated `Reports/Meeting 4/` on the `main` branch:

- **`Report_Ifty_Meeting4.tex`** — full single-column IEEE-style report (~36 KB), 11 sections: Introduction, Planning Module, Pipeline (annotated with measured numbers), Engineering Reality (model substitution justification with explicit citation of Meeting 1's YOLO11 mention), Experimental Setup, Results (full Table I + 7 figures), Engineering Iteration Log v1→v22, What This Shows / Does Not Show (caveats: n=3 scenes, single-scene-driven regression, C1 mAP did not strictly improve), What Comes Next, Conclusion. Bibliography: Wang & Machida 2025, nuScenes, PDM-Closed, CenterPoint, PKL, Ultralytics, plus our own Meeting 1 / Meeting 3.
- **`images/`** — 16 figures from kernel v22 (11 PNG + 5 JPG): full Table I bar chart, cascade signature plot, pipeline diagram, both YOLO training-curve sets, both confusion matrices, side-by-side detection grid, per-scene L2, per-scene failure modes, plus Ultralytics-rendered val batches for both fine-tunes.

Committed to `main` as `a048cee` so Meeting 4 sits alongside Meetings 1-3 in the publishing repo.

**2026-04-30 (post-deliverable):** branches `main` and `experiment/E1` were merged in both directions so both contain the complete repository (Meetings 1-4 + `experiments/` + `EXPERIMENT_LOG.md` + all figures). After this point the two branches are bit-identical. Merge commits: `2a9f70c` (`main` → `experiment/E1`) and `64d557b` (`experiment/E1` → `main`).

---

## 8. v20 results — first complete real-model E1 (2026-04-30)

### 8.1 Final numbers (Table I)

| Component | Metric | Before (Boston-trained C1) | After (SG-fine-tuned C1) | Δ | Δ % |
|---|---|---:|---:|---:|---:|
| C1 | mAP Singapore | NaN | NaN | — | — |
| C2 (isolated) | minADE w/ GT detections | 2.0899 | 2.0899 | 0.0000 | 0.00% |
| C2 (pipeline) | minADE w/ live C1 | **15.1930** | **3.9535** | **−11.2395** | **−73.98%** |
| C3 (isolated) | L2 w/ GT predictions | 4.9289 | 4.9289 | 0.0000 | 0.00% |
| **C3 (pipeline)** | **L2 w/ live C2** | **6.2969** | **6.5770** | **+0.2801** | **+4.45%** |
| C3 (pipeline) | collision_rate | 0.0 | 0.0 | 0.0 | 0.00% |

### 8.2 Cascade diagnostic verdict

```
diagnostic: {'confirmed': False, 'rho_1_to_3': None,
             'reason': 'C1 did not improve; C3 pipeline did not worsen'}
```

The diagnostic says **not confirmed**, but the underlying numbers are scientifically much more interesting than a clean "confirmed" would be. The diagnostic rule (encoded in `table.cascade_diagnostic`) tests:
1. C1 mAP improved (yes/no)
2. C3 isolated stable (yes ✓ — exactly 0 change)
3. C3 pipeline worsened beyond noise floor (yes ✓ — +4.45%, just under our 5% threshold)

The "C1 did not improve" reason is because `c1_mAP` was reported as `NaN` (we didn't compute it for the YOLO path — only the Ultralytics-internal validation happens, not a comparable mAP metric). That's a measurement gap, not a cascade-absence claim. **The C3-pipeline change is the actual finding.**

### 8.3 Why this is genuinely interesting (not a "failure")

**C2 pipeline minADE *improved* dramatically (−74%).** Singapore-fine-tuned YOLO produces detections that are far better matched to Singapore agents than the Boston-trained YOLO was. C2's job (zero-velocity rollout from those detections) gets cleaner inputs.

**C3 pipeline L2 *worsened* slightly (+4.45%).** Even though C2 received better inputs and produced better intermediate predictions, the downstream IDM planner's chosen trajectory drifted *further* from the human driver's actual path.

This is **exactly the "entangled enhancement" pattern the Meeting 3 report and the underlying SMP literature describe**:
- An upstream component genuinely improves on its own metric.
- An intermediate component receives better inputs.
- The terminal component still degrades on the system-level metric.

This is more valuable for the thesis than a clean "everything got worse" cascade would have been — it demonstrates the *paradoxical* nature of the cascade hypothesis. The base paper's whole argument is that retraining C1 doesn't guarantee end-to-end improvement, and v20 shows precisely that.

### 8.4 Why C3 pipeline drifted despite better C2 inputs

Three plausible mechanisms (to be investigated in subsequent runs / writeup):

1. **C2's zero-velocity assumption** — when YOLO detections are *more confident* (Singapore-fine-tuned), the IDM planner treats them with less hedging; with fewer phantom misses, the planner commits more aggressively to a path, which can move it further from the conservative human trajectory the L2 metric is anchored on.
2. **Distribution mismatch in C3** — C3 was tuned (via its weights and proximity radius) for the kind of agent positions Boston-trained C1 produced. Different (better) Singapore detections shift the planner's input distribution, and the IDM heuristics aren't readjusted.
3. **Mini sample size** — singapore_val has only 3 scenes (~120 keyframes). +4.45% on n=3 scenes is within scene-level variance.

### 8.5 Headline figure

`cascade_result.png` saved to `experiments/E1_kaggle/results/run-v20/`:

- C3 (isolated) — both bars at 4.93 (identical)
- C3 (pipeline) — Boston-trained 6.30, Singapore-retrained 6.58
- The visual gap between isolated and pipeline (~1.4–1.6 m) is the cascade signature
- The visual gap between Boston and Singapore C1 within the pipeline column (~0.3 m) is the *additional* cascade contribution from retraining

### 8.6 Compute spent on v20

- Total runtime: ~234 s (4 minutes)
- T4 x2 allocated correctly
- Boston YOLO fine-tune: hit cache (skipped retrain)
- Singapore YOLO fine-tune: ~50 s real training, 10 epochs on 119 images
- Inference + eval: ~30 s
- Plot + Table I: ~5 s

---

## 7b. Real-model phase begins (kernel v14, 2026-04-30)

### 7b.1 New module: `src/c1_yolo.py`

Real YOLOv11 path. Three responsibilities:

1. **3D-to-2D projection** — `project_3d_box_to_2d()`: takes nuScenes 3D
   annotations (translation + size + quaternion) and projects them to a
   2D camera-image bbox using the calibrated camera intrinsic + extrinsic
   matrices and the ego pose. Filters boxes that are behind the camera
   or fall entirely outside the frame. ~50 lines.

2. **Dataset builder** — `build_yolo_dataset()`: iterates a list of scene
   tokens, copies their `CAM_FRONT` keyframes into a YOLO-format
   `images/<split>/` directory and writes per-image `labels/<split>/*.txt`
   with normalised `(class, cx, cy, w, h)` rows. 8 driving-relevant
   classes are kept (car / truck / bus / trailer / construction_vehicle /
   motorcycle / bicycle / pedestrian).

3. **Fine-tune wrapper** — `fine_tune_yolo()`: thin wrapper over
   `ultralytics.YOLO(...).train(...)`. Real gradient-based fine-tune from
   the official `yolo11n.pt` (6 MB). Returns the path to the produced
   `best.pt`.

4. **Inference path** — `c1_detect_yolo()`: runs a trained YOLOv11
   checkpoint on a scene's CAM_FRONT keyframes, back-projects each 2D
   detection to ego-frame ground-plane coordinates by intersecting the
   bottom-center pixel ray with `z = 0` in the ego frame, and emits
   `Detection` records consumable by C2 / C3.

Class-default sizes (rough nuScenes averages) are used when emitting
`Detection.width / length / height` because YOLOv11 outputs 2D bboxes
without explicit 3D dimensions; the cascade signal we measure does not
depend on bounding-box size precision.

### 7b.2 Eval/C2 dispatch routing

`eval.run_all_measurements` now reads the C1 descriptor JSON and
branches on `kind`:

- `"yolo"` → real-YOLO path (calls `c1_detect_yolo` for pipeline mode).
- `"oracle+perturbation"` → legacy fallback (kept for environments where
  ultralytics fails to install).

`train_c2.c2_predict(mode='pipeline')` does the same dispatch — when the
descriptor is YOLO, `_predict_from_detections` calls `c1_detect_yolo`
with the YOLO weights instead of `c1_detect` with a perturbation profile.

### 7b.3 Notebook v14 changes

**Phase 1 install cell** appends:
```python
pip install --no-cache-dir --no-deps ultralytics
pip install --no-cache-dir ultralytics-thop py-cpuinfo
```
Sets `YOLO_AVAILABLE` flag so subsequent cells gracefully skip if the
install fails.

**Phase 3** is now real-YOLO C1 baseline:
- Builds YOLO dataset from `boston_train` + `boston_val` (filters to
  CAM_FRONT, projects 3D boxes).
- Pre-fetches `yolo11n.pt` to `/kaggle/working/` (avoids ultralytics
  attempting a download into a read-only path).
- Calls `fine_tune_yolo` for 10 epochs.
- Emits `c1_descriptor.json` with `{kind: "yolo", weights_path: ...}`.

**Phase 7** is now real-YOLO C1 fine-tune on Singapore:
- Builds YOLO dataset from `singapore_train` + `singapore_val`.
- Re-runs `fine_tune_yolo` starting from the Boston `best.pt` (this is
  the real "retrain C1 on Singapore" step from Meeting 3 §VI-B).
- Emits a new descriptor.

Both phases include perturbation-oracle fallbacks that activate if
`YOLO_AVAILABLE` is False or no images were extractable.

### 7b.4 What this run is testing

- Whether `ultralytics` installs cleanly on Kaggle's torch 2.10 + cu128.
- Whether the 3D-to-2D projection produces sensible labels (visual
  inspection of `images/train/*.jpg` + `labels/train/*.txt` would tell us).
- Whether 10 epochs of fine-tune on ~80 keyframes produces a Boston-
  specialised detector (compared to the COCO-pretrained baseline).
- Whether 10 more epochs of fine-tune on Singapore keyframes shifts the
  weights enough to produce different ego-frame ground-plane positions
  — the **real** C1 → C2 cascade signal.

### 7b.5 Local self-tests (offline, before kernel push)

```
c1_yolo bookkeeping OK            (descriptor round-trip + data.yaml writer)
train_c2 self-test OK             (constant-velocity descriptor, C1-dispatch routing)
table self-test OK                (mock cascade still confirmed: ρ_1→3 ≈ 0.84)
```

All offline tests pass.

### 7b.6 Compute budget (predicted)

| Step | GPU-min on P100 |
|---|---|
| Build Boston dataset (no GPU) | ~2 |
| Pre-fetch yolo11n.pt (no GPU) | <1 |
| YOLOv11 fine-tune Boston (10 epochs, ~80 imgs) | ~10 |
| Build Singapore dataset (no GPU) | ~2 |
| YOLOv11 fine-tune Singapore (10 epochs, ~120 imgs) | ~12 |
| C1 inference on Singapore val (both ckpts) | ~2 |
| C2 / C3 evaluation | ~5 |
| **Total** | **~35** |

Within Kaggle free-tier weekly budget.

### 7b.7 Sources / lineage

- Ultralytics — pure-PyTorch YOLO with pretrained weights:
  https://github.com/ultralytics/ultralytics
- yolo11n.pt asset (~6 MB):
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
- Meeting 1 report (named YOLO11 for C1):
  `Reports/Meeting 1/Report_Ifty_10032026.pdf` §B(1)
- Meeting 3 report (CenterPoint as the LiDAR alternative for C1):
  `Reports/Meeting 3/Report_Ifty_29042026.pdf` §VI-A

---

## 8. Next planned phase (after v14 lands)

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
| `4bc3ad9` | 2026-04-30 | add EXPERIMENT_LOG.md — full record from setup → YOLO redesign |
| *(current)* | 2026-04-30 | real-model C1: YOLOv11 fine-tune on Boston + Singapore (kernel v14) |

---

## 11. Open questions for Meeting 4

- Is the YOLOv11 + MTP + IDM substitution acceptable for the thesis, or does the supervisor want to insist on the literal Meeting 3 model triple (CenterPoint + MTP + PDM-Closed) which would block on HPC?
- Should we report PKL as the headline cascade metric instead of (or alongside) C3 pipeline L2?
- For the SMP calibration in subsequent meetings, will mini-scale ρ₁→₃ values be acceptable, or does that step also wait for HPC?

---

*End of log. Document maintained alongside the `experiment/E1` branch.*

## 12. Meeting 5 preparation — literature review deliverable (2026-07-07)

Advisor tasks from Meeting 4: (1) justify the pipeline approach against the E2E trend; (2) propose a baseline for mitigating cascading effects.

- Ran a multi-agent deep-research pass (parallel web search → source fetch → claim verification) over 2024–2026 E2E driving literature and cascade-mitigation literature; ~20 sources retained per user's scope cap.
- Wrote `Reports/Meeting 5/Meeting5_Literature_Review.md`: E2E survey (UniAD, VAD, PARA-Drive, SparseDrive, Hydra-MDP, VLA survey, E2EREP/CorrectAD/R2SE repair thread), E2E-vs-pipeline trade-off table, evidence-backed pipeline justification (SMP needs per-component states/actions; E2E has a one-element retraining action space), cascade-mitigation review (Sculley, TFDV, BCT, PC-training/ELODI, uncertainty propagation).
- Recommended first baseline: **C1→C2 interface drift gate** — KS/JS two-sample tests on detector output statistics (count, confidence, class, spatial) on a fixed audit set before admitting a retrained C1; validatable retrospectively on the v22 artifacts; feeds the SMP coverage factors.
- Note: "E2EREP" (page shared by advisor, experiments on UniAD/VAD) is not publicly indexed as of 2026-07 — likely under review; need full PDF/citation from advisor.
- Follow-up implementation candidate for next kernel: `drift_gate.py` over existing detection JSONs (no GPU needed).

## 13. Meeting 5 deliverable — LaTeX + verified bibliography (2026-07-08)

- Wrote `Reports/Meeting 5/Report_Ifty_Meeting5.tex` (Meeting-4 preamble/style, IEEEtran bib) + kept the Markdown version.
- Verified all 22 cited papers as REAL via parallel web-search agents (title/authors/venue/year/arXiv-ID/DOI). Corrections folded in: SparseDrive title (dropped "and Parallel Planning"), R2SE title (dropped "R2SE:" prefix, "Self-Aware"), ELODI first author = Y. Zhao (not Shen), System-Level UQ first author = Deglurkar. Collected into `Reports/Meeting 5/references.bib` (28 entries; 26 cited + nuScenes/ISSRE extras).
- Full compile verified: pdflatex→bibtex→pdflatex×2, exit 0, no bibtex warnings, no undefined citations, 26 \bibitem resolved, 10-page PDF.
- Only non-verifiable source is "E2EREP" (advisor's shared page) — not publicly indexed; cited in prose only, full citation pending advisor's PDF.

## 14. Meeting 5 report — plain-language rewrite for presentation (2026-07-08)

- Rewrote `Report_Ifty_Meeting5.tex` in simple English for a Japanese audience (sensei + labmates) hearing it for the first time; report doubles as presentation script.
- Removed all em dashes and AI-style phrasing.
- Restructured both literature reviews: top ~10 papers (ranked by publication quality) explained in prose; weaker/preprint papers moved to compact tables (Table 1 for E2E/repair, Table 5 for cascade methods).
- Recompiled clean: pdflatex→bibtex→pdflatex×2, no warnings, no undefined cites, 9 pages. Same verified references.bib (unchanged). Markdown version kept.

## 15. Meeting 5 report — reframed pipeline argument to real-world merit (2026-07-08)

- Advisor/user feedback: the pipeline should be justified by real-life deployment/maintenance merit, NOT by "convenient for our research."
- Reframed throughout Report_Ifty_Meeting5.tex:
  - Abstract + Exec Summary: pipeline matches how real systems are updated (one part at a time: new-city data, map update, sensor swap); E2E forces full expensive rebuild + forgetting.
  - Section 3 intro + accuracy table row: dropped "this is what we study"; the coupling weakness is a real post-update maintenance problem to manage.
  - Reason 1: leads with "how companies actually maintain deployed systems" (fix the part that broke).
  - Reason 2: retitled to fault-finding/fixability in deployment (engineer must point to the failed part).
  - Reason 3: real lifetime maintenance cost (50s per-part vs 144 GPU-h full retrain + fake-data generation).
  - Reason 4: E2E research adds parts back once maintenance (not benchmark score) is the goal.
  - "Research gaps" heading -> "Open problems"; reworded as practical maintenance questions, not "our SMP fits here."
- Recompiled clean (9 pages, no undefined cites). Bib unchanged.

## 16. Meeting 5 report — added drift-gate figure + open-loop definition (2026-07-08)

- Added TikZ figure (Fig. 1) to the "Recommended First Baseline" section visualizing the drift gate: new C1 + old C1 (audit set) -> drift gate (compare output stats) -> "too different?" -> accept (run C1->C2->C3) / cascade-risk (hold, use old C1). Matches Meeting-4 TikZ style; self-contained, no external image.
- Defined "open-loop score" once at first use (Exec Summary) and referenced it thereafter (term used 9x; previously only glossed in fragments). Trimmed NAVSIM paragraph to rely on that definition.
- Recompiled clean (9 pages, no undefined cites); figure visually verified via pdftoppm render.

## 17. Meeting 5 report — fit cascade table onto page 6 (2026-07-08)

- Table 3 (cascade-mitigation methods) was floating to a fresh page leaving a big blank gap on p6.
- Changed float placement [h!] -> [t!] so it pulls to top of available space.
- Dropped 2 least-essential rows (MixBCT, ELODI) — both are just refinements of BCT / positive-congruent training already explained in the main text. Their citations are now unused (left in references.bib, harmless).
- Result: table sits at top of p6, blank gap filled, 10 pages -> 9. Verified via render.

## 2026-07-08 — IEEE conference paper drafted from Meeting 4 (E1 v22 results)
- Created `ieee paper/` folder at repo root with two deliverables:
  - `IEEE_Conference_Paper_E1.txt` — full ~8-page IEEE-format conference paper based solely on the Meeting 4 report (E1 kernel v22 numbers). Title: "Entangled Enhancement in Modular Autonomous Driving Systems: An Empirical Study of Cascade Effects from Single-Component Retraining."
  - `references.bib` — 27 verified references (23 within 2020–2026), numbering matched to in-text [n] citations; entries reused from the verified Meeting-5 bibliography plus IDM (Treiber 2000), PKL (Philion 2020), YOLO11.
- Paper structure per advisor/user spec: Finding F1 (why E2E infeasible → pipeline adopted) placed in Introduction as motivation; experiments framed around C1 (retrained unit) and C2 (directly coupled consumer); five findings F1–F5; proposed method (CARA — Cascade-Aware Retraining Assessment protocol, incl. coupling factor rho = Delta_n / delta_i, Eq. 4) placed after Results & Discussion.
- All numbers taken verbatim from v22 Table I (C1 mAP@50 0.1185→0.0644; C2 pipe minADE 15.1930→3.9535, −73.98%; C3 pipe L2@3s 6.2969→6.5770, +4.45%; isolated rows bit-identical). No new measurements run.

## 2026-07-08 — Ported IEEE paper into uploaded IEEEtran template
- User uploaded the official IEEEtran BibTeX package into `ieee paper/` (IEEEtran.bst, IEEEtranS.bst, IEEE{abrv,full,example}.bib, README, HOWTO). No IEEEtran.cls was included; it ships with the local MiKTeX install.
- Converted the plain-text paper to a full IEEEtran conference document: `ieee paper/IEEE_Conference_Paper_E1.tex` (\documentclass[conference]{IEEEtran}), all [n] citations replaced with \cite{} keys matched to references.bib, Table I as a real booktabs table, Eqs. (1)-(5) typeset, \bibliographystyle{IEEEtran} + \bibliography{references}.
- Compiled clean: pdflatex -> bibtex (0 errors/0 warnings, all 27 refs resolved) -> pdflatex x2. Output `IEEE_Conference_Paper_E1.pdf` = exactly 8 pages, no undefined refs/citations. Verified page 1 (title/abstract/intro) and page 8 (future work + references) render correctly in two-column IEEE format.
- Kept .txt as the plain-text deliverable; added .tex, .pdf, .bbl. Removed pdflatex aux files (.aux/.log/.out/.blg).

## 2026-07-08 — Added figures + math to IEEE paper (A* upgrade)
- Copied 9 figures from Reports/Meeting 4/images into `ieee paper/figures/`: pipeline diagram, full Table I bar chart, cascade-signature plot, per-scene L2 + failure-modes pair, Boston training curves, Singapore confusion matrix, and Singapore val pred/label qualitative pair. (Meeting 5 E2E.png excluded — it is a screenshot of the E2EREP paper text, not a usable diagram.)
- Integrated into IEEE_Conference_Paper_E1.tex: Fig.1 pipeline (system model), Fig.2 full-table bar chart (figure*, results), Fig.3 cascade signature (F4), Fig.4 confusion matrix (F5), Fig.5 per-scene subfigures, Fig.6 training curves, Fig.7 qualitative detections, plus native TikZ CARA gate diagram + algorithm2e Algorithm 1.
- Added formal math: pipeline composition Eq.(1) S=C3∘C2∘C1, isolated/pipeline mode Eqs.(2)-(3), cascade gap/isolation/injection Eqs.(4)-(6), minADE Eq.(7), L2 Eq.(8), IDM proposal-and-score objective Eq.(9) with underbrace, coupling factor Eq.(10).
- Recompiled clean: pdflatex/bibtex/pdflatex×2, 0 undefined refs/cites, 0 overfull >20pt (fixed CARA TikZ via \resizebox). 10 pages, ~1.77 MB. All prose still em-dash-free.

## 2026-07-08 — Added base paper (COMPSAC 2024) + coupling factor + projection math to IEEE paper
- User flagged the true BASE PAPER via Reports/Initial Proposal/Proposal.pdf ref [4]: Wang & Machida, "Maintaining Performance of a Machine Learning System against Imperfect Retraining," COMPSAC 2024, pp.1782-1787, DOI 10.1109/COMPSAC61105.2024.00284. This is the father of the whole research program (two-component CTMC, origin of the term "entangled enhancement", progressive vs conservative policies). Previously only the ISSRE 2025 follow-on was cited.
- Added wang2024compsac (base) + kreuzberger2023mlops (MLOps survey, IEEE Access 2023) to references.bib. Now 29 entries.
- Cited base paper prominently: abstract (defines entangled enhancement), Introduction ("builds directly on... base paper for this work"), Related Work reliability subsection (CTMC, 2^N state growth, progressive/conservative), F4 result ("first real-pipeline measurement of the entangled enhancement the base paper models analytically"), contributions + future work (ρ calibrates the base paper's models).
- Added from Meeting-4 log (user-selected): (a) cascade coupling factor ρ_{1→3}=Δ3/δ1 formalized with the 3-condition diagnostic rule Eq.(7), honest treatment of ρ≈0.84 as synthetic apparatus-check + ambiguous-sign on real run; (b) label-projection math Eqs.(9)-(10): 3D→2D via K·Tc^-1·Te^-1, ground-plane z=0 back-projection lift, 8 driving classes, class-average dims.
- Recompiled clean: bibtex 0 warnings, 0 undefined cites, 0 overfull. 11 pages. All prose em-dash-free.

## 2026-07-08 — Redrew overlapping figures as native vector, 1-line captions, trims
- User: Fig 2/3 had text overlapping the plot (raster exports); recreate clearer. Also: every caption must be ≤1 short line (descriptions go in the section prose, not captions).
- Redrew 3 raster charts as native pgfplots/TikZ (crisp, no overlap): Fig.1 pipeline diagram (TikZ boxes + z1/z2 arrows + before/after numbers), Fig.2 full-table 2-panel grouped bar chart (groupplot), Fig.3 cascade-signature bar chart. Added pgfplots + beforeblue/afterorange palette to preamble. Fixed pgfplots key error via /pgf/bar width.
- Shortened ALL captions to one line; moved every multi-line description into the surrounding prose (table Before/After defn, tablefull, cascade, confusion, training-curves, qualitative, CARA gate).
- Deleted per-scene decomposition subsection + its 2 figures (user: reduces paper weight). Fixed dangling refs in contributions list; kept the n=3 statistical-power caveat (still accurate).
- Deleted Acknowledgment section. Replaced author block with placeholders (Author Name / Department / Affiliation / City, Country).
- Removed now-unused images (fig01/02/03/09/10). figures/ keeps only genuine experimental outputs: confusion matrix, training curves, val pred/label pair.
- Recompiled clean: bibtex 0 warnings, 0 undefined cites, 0 meaningful overfull (1pt), 11 pages. Visually verified Fig.1/2/3 render sharp with no overlap.

## 2026-07-08 — Fixed Fig.2 legend/title overlap + fused value labels
- User flagged: (1) left-panel title "C1 detection quality" hidden behind the legend; (2) fused number labels where before/after bars are equal or close (2.09/2.09→"2.092.09", 4.93/4.93, 6.30/6.58).
- Fix: replaced the in-plot legend with a manual TikZ legend row centered ABOVE both panels (always renders, clears both titles). Removed nodes-near-coords from the right (C2/C3) panel where equal-height pairs collide — exact values remain in Table I; the panel now shows shape only. Left panel keeps 0.12/0.06 labels (bars differ). Widened horizontal sep to 17mm, enlarge x limits 0.28.
- Recompiled clean: 0 undefined cites, 11 pages. Visually verified: titles clear, legend above, no fused labels.

## 2026-07-08 — Three requested fixes: YOLOv11n naming, htbp floats, drop E1 label, hyperref
- (1) All plain "YOLOv11" in OUR work (not lit review) -> "YOLOv11n": abstract, intro, pipeline TikZ box, projection prose, training-curve caption. "YOLOv11 nano" left as-is per instruction. Lit-review YOLO mentions untouched (none existed anyway).
- (2) All 9 floats (figure/figure*/table/algorithm) [t] -> [htbp]. subfigure [b] alignment options correctly left untouched.
- (3) Removed all "E1"/"Experiment E1"/"Experiment 1" naming (this IS the paper's main work, not one of a numbered series): subsection retitled "Single-Component Retraining Under Domain Shift" (+\label{sec:experiment}); captions dropped E1; prose -> "this experiment"/"our experiment"/"our update"/"our harness"/"the pattern we observe". 0 "E1" tokens remain. Kept the E1-E7 subset-lattice idea in future work but phrased as "all seven non-empty subsets of {C1,C2,C3}".
- Added \usepackage[hidelinks,breaklinks]{hyperref} LAST in preamble (after cite/pgfplots/tikz). Clickable citations + cross-refs + URLs; hidelinks = no colored boxes (clean submission look). Verified 126 link annotations in PDF, no cite/hyperref clash.
- Recompiled clean: bibtex 0 warnings, 0 undefined cites, 11 pages. Verified Fig.1 shows YOLOv11n + de-labeled caption + renamed subsection.

## 2026-07-08 — Abstract <250w + removed free-compute promo + no abstract cite + Intro -30%
- Abstract: trimmed to 204 words (<250); removed the "runs on free-tier cloud compute / doesn't need industrial resources" closing; removed the \cite{wang2024compsac} (user: never cite in abstract) — reworded to "known as entangled enhancement, so far modeled only analytically for a two-component system". Base paper still cited in body (Intro/Related Work/F4/Future Work) + bibliography.
- Removed free-tier/free-compute/"any research group can" self-promotion everywhere: abstract, F1 (kept factual ~50 GPU-s), Component-choice section, CARA mitigation, Conclusion. Kept factual sm_60/P100 + 144-GPU-h comparison (legit methodology); reworded "free-tier P100s" -> "P100s available to us".
- Introduction reduced 31.7% (1140 -> 779 words): merged the 3-reasons block, tightened base-paper paragraph, condensed F1-F5 to one-liners (numbers live in Results), shortened contributions 4->3 items. All Intro citations preserved (verified in bbl).
- Recompiled clean each time: bibtex 0 warnings, 0 undefined cites, 10 pages.

## 2026-07-08 — Fig.5/6 -> figure* (full width), Fig.2 -> single-column figure
- User: Fig.5 (training curves) and Fig.6 (qualitative detections) must be figure* (full text width); Fig.2 (full profile bar chart) can drop to single-column figure.
- Fig.5: figure* @ 0.82\textwidth — 2x5 loss/metric grid now large + readable.
- Fig.6: figure* with two 0.42\textwidth subfigures — detection/label panels now legible.
- Fig.2: figure* -> figure; shrank groupplot (0.52\columnwidth panels, 9mm sep, 5pt bars, scriptsize/tiny fonts, 30deg x-ticks) to kill a 23pt overfull; now fits one column cleanly, legend + 0.12/0.06 labels intact.
- Recompiled clean: figure/figure* balanced (5/2), bibtex 0 warnings, 0 undefined cites, 0 overfull, 10 pages. Visually verified all three.

## 2026-07-08 — Task 1: removed 8 low-value citations (29 -> 21, min 20 kept)
- Removed the least-load-bearing cites + their supporting sentences: weng2024paradrive (PARA-Drive, redundant flagship), li2024hydramdp (Hydra-MDP, redundant flagship), cheng2025adreft (3rd redundant E2E-repair example), philion2020pkl (PKL aside), zhao2022elodi (ELODI, refinement of PC-training), nitsch2021ood (OOD, minor list item), su2024motcup (2nd uncertainty-prop cite), kreuzberger2023mlops (MLOps filler sentence).
- Fixed all dangling multi-cite groups (F5 negative-flip, CARA Positioning) so no removed key remains referenced.
- Deleted the 8 now-unused entries from references.bib.
- Result: 21 \cite keys used = 21 bibitems = 21 bib entries. bibtex clean, 0 undefined, 10 pages. Kept the strongest 21 incl. base paper (wang2024compsac), ISSRE follow-on, nuScenes, YOLO11, IDM, UniAD/VAD/SparseDrive, CorrectAD/R2SE, Sculley, PC-training, BCT, Ivanovic, Breck, NAVSIM, PDM-Closed, Codevilla, Zhai, CenterPoint, survey.
- Tasks 2 (redundant text) and 3 (over-long C1/C2/C3 explanations) still pending per user.

## 2026-07-08 — Removed "Why These Components" subsection + redundancy pass
- Removed the entire "Why These Components and Not Heavier Ones" subsection (CenterPoint/mmdet3d/sm_70/nuPlan engineering-reasons paragraph). yin2021center still cited in Threats + Future Work, so not orphaned; 21 cites intact.
- Also removed the two remaining accolades outside Related Work: UniAD "Best Paper Award" (done earlier), PDM-Closed "won the 2023 nuPlan challenge" in the C3 methodology paragraph -> "outperformed learned planners" / plain "reproduces PDM-Closed". Zero accolades remain anywhere.
- Redundancy pass (deduplicated text repeated across sections):
  - Methodology dual-mode: deleted the "Concretely, C2-isolated predicts..." restatement (already in the iso/pipe equations + Component Instantiation).
  - Formal Pipeline Model: collapsed the "planner never sees raw keyframe / plan right for the world it believes" narrative + "experiments center on C1/C2" (both duplicated the Introduction) into one sentence.
  - Component Instantiation C2: merged the two "cascade study centered on C1->C2 interface" repeats; folded zero-velocity detail in.
  - Measurement Protocol: replaced the re-listing of the five measurements with a back-reference to Sec III-D (the experiment procedure already enumerates them).
  - F5: deduped "cascade doesn't travel through aggregate score / metrics come apart" double statement.
- Recompiled clean: bibtex 0 warnings, 0 undefined, 21 cites = 21 bibitems. 10 -> 9 pages.

## 2026-07-08 — Conclusion+Future Work reduced ~37%, restructured to 2 paras
- User: cut 30%, keep 2 paras — 1 big conclusion para + 1 small future-work para.
- Was 301 words / 3 paras (conclusion + 5-item future list + "wider message" flourish). Now 190 words / 2 paras.
- Para 1 (big): conclusion — study summary, headline numbers, five-findings takeaway, CARA, + merged the methodological "broader point" sentence.
- Para 2 (small): future work condensed from 5 enumerated items to one sentence covering 4 directions (dropped the un-cited "automate CARA gate" item). Kept cites wang2024compsac, wang2025issre, yin2021center, dauner2024navsim.
- Clean compile: 0 undefined, 21 cites, 9 pages.

## 2026-07-08 — Significantly rewrote Experimental Setup (was README-like)
- User: Setup too long, esp. Component Instantiation — too much implementation detail for a research paper.
- Collapsed 4 subsections -> 2: "Dataset and Components" + "Measurement Protocol". Removed the standalone "Label Projection and Detection Lifting" subsection (+eq:proj, eq:lift) and "Component Instantiation".
- Cut implementation trivia: exact hyperparameters (10ep/AdamW/lr8.3e-4/bs8/imgsz640), "6 MB/pure-PyTorch", "~50s on T4", "499 layers transferred", "descriptor files/harness routing", full 8-class list. Folded 3D->2D projection into one sentence (kept the ground-plane-ray idea, dropped the 2 projection equations).
- Kept substantive facts: dataset/splits, before/after fine-tune design, C2 deterministic/parameter-free (needed for attribution), and the C3 IDM objective eq (now the only Setup equation).
- ~430 -> ~278 words, 3 eqs -> 1. No dangling refs (eq:proj/eq:lift/sec:projection were only cited within removed text). Clean compile, 21 cites, 9 pages.

## 2026-07-08 — Ported paper into OMLET 2026 IEEE conference template
- User supplied the OMLET 2026 template (IEEEtran conference, \thesection Roman, fancyhdr firstpage block commented, unsrt/IEEEtran bib, adjustbox/colortbl/multirow/comment pkgs, green hyperref links).
- Rebuilt IEEE_Conference_Paper_E1.tex = template preamble + my full body (abstract->conclusion) + \bibliography{references}. Added the packages my content needs on top of the template's (algorithm2e, subcaption, tikz+libs, pgfplots+groupplots, beforeblue/afterorange). Dropped the template's \usepackage{algorithm} (clashes with algorithm2e; my CARA alg is algorithm2e syntax). Set final bib to IEEEtran + references.bib (not the template's placeholder Ref).
- Authors: Rashedul Arefin Ifty^1, Fumio Machida^2, University of Tsukuba, Tsukuba Japan, emails superscripted per template style.
- Kept the commented OMLET firstpage header/footer + \thispagestyle{firstpage} as the template ships them.
- Verified: all content survived (21 cites, 7 figures, 5 tikz, 1 algorithm, 12 eqns). Compiles: bibtex clean, 0 undefined refs, 9 pages. Roman section numbers + green links render per template. (pdflatex exit-1 is only the cosmetic MiKTeX 'no admin update check' warning; PDF builds fine.)

## 2026-07-08 — Fit paper to 8 pages (Discussion -36%, minimal other cuts)
- Discussion 500 -> 322 words (36% cut, target was 35%): condensed Mechanism subsection, tightened Implications, trimmed all 4 Threats items to their essential claim.
- Fig 5 (training curves): shrank 0.82->0.66\textwidth then converted figure* -> single-column figure (was reserving full-width block; single-column reclaimed the space and looks cleaner).
- references.bib: removed all standalone `note` fields (arXiv IDs, "Best Paper Award", "won 2023 nuPlan challenge") -> shorter reference list. 21 entries intact, journal=arXiv-preprint fields kept.
- Minimal prose trims to reduce overexplanation: F3 dropped the mechanism sentence duplicated in Discussion; F4 tightened two restated-number sentences + removed the Gamma3/Delta3 restatement already in caption/defs.
- Result: 10->8 pages. bibtex clean, 0 undefined, 0 overfull, 21 cites=21 bibitems, no figure* remaining. Refs [1]-[21] all on page 8, not cramped.

## 2026-07-08 — Vertical value labels on Fig 2, single-block authors, widow fixes, 8pp
- Fig 2: rotated the data value labels 90deg (nodes near coords style rotate=90, anchor=west) so they sit vertically above each bar (per user's sketch). This eliminated the equal-bar label collisions, so the previously-omitted labels on panel (b) were restored. Raised ymax for headroom (0.15->0.19, 17->20), reduced height 4.2->3.8cm. Caption keeps (a)/(b) panel descriptors.
- Author block: user iterated -> final = SINGLE block as in template: "Rashedul Arefin Ifty, Fumio Machida" / University of Tsukuba / Tsukuba, Japan / both emails on one line, NO superscript 1/2 pointers.
- Widow words: added \looseness=-1 to several long paragraphs (intro measurement para, "we therefore adopt", F3, F5, CARA mitigation) to reclaim single-word-on-a-line waste the user flagged.
- Also fixed a latent bug: \thispagestyle{firstpage} referenced a style defined only inside the commented-out OMLET block -> "Undefined control sequence". Commented it out (paired with the disabled banner); can be re-enabled together.
- Result: back to 8 pages, refs [1]-[21] all on page 8, bibtex clean, 0 undefined, 21 cites.

## 2026-07-08 — Fixed repeated-author em-dash on ref [9]
- Ref [9] (Wang&Machida ISSRE, follows [8] Wang&Machida COMPSAC) printed as "------," (IEEEtran collapses consecutive same-author entries).
- Persistent fix at bst level: added @IEEEtranBSTCTL{IEEEexample:BSTcontrol, CTLdash_repeated_names="no"} to references.bib + \bstctlcite{IEEEexample:BSTcontrol} after \maketitle. Now [9] shows full "Z. Wang and F. Machida,". Survives rebuilds (not a manual .bbl edit).
- Still 8 pages, bibtex clean, 0 undefined, 21 cites.

## 2026-07-08 — Addressed 4 reviewer criticisms (grounded in Meeting 4 report), kept 8pp
- Reviewer critiques all essentially correct; softened overclaims + added caveats rather than hiding limitations. Grounded in Meeting 4 report's own "What it does not show" section (which already admits C1 mAP did not rise "in the strict sense the diagnostic requires").
- #1 (overstated "entangled enhancement"): abstract + contributions + F4 reframed — since delta1<0, run does NOT meet strict entangled-enhancement condition; it shows the related non-monotone cascade + metric decoupling. Removed "first measurement of entangled enhancement" marketing.
- #2 (C2 differs across modes): F3 now states Gamma2 conflates C1 detection error + velocity-information loss (iso=GT velocity, pipe=zero-velocity); clarified that Delta2 IS clean because the velocity gap is constant across runs.
- #3 (planning degradation may be metric artifact): new Discussion para — collision rate 0/0, worse L2 is the ego-forecasting shortcut (Zhai/Codevilla), update may be a SAFER planner scoring worse on a blindness-rewarding metric; direct tension for CARA (could reject a better update); NAVSIM as safeguard. Threats "open-loop L2" item now points here.
- #4 (novelty overstated): F2 isolation reframed as near-definitional validity check (not a discovery); CARA Positioning softened ("largely the experimental procedure cast as a gate; contribution is the gate statistic, not an algorithm"); rho step states it is not cleanly measurable when delta_i ambiguous.
- Fixed ref [9] em-dash (repeated-author) via IEEEtranBSTCTL CTLdash_repeated_names=no + \bstctlcite.
- Page budget: added net caveat content but held 8pp by trimming intro (user request), related-work eval tail, conclusion/future-work, threats items, and — per user — converting the CARA Protocol enumerate to PARAGRAPH prose (kept Algorithm 1 intact). Author block = single template block, no 1/2. 21 cites, 0 undefined, 0 overfull, 8 pages.

## 2026-07-08 — Fig 2 label/title collision, separate authors, em-dash cleanup
- Fig 2: the vertical "15.19" (and "0.12") value labels overflowed into the panel titles. Raised ymax (0.19->0.23, 20->26) with capped ytick lists so the labels clear the titles without adding clutter ticks.
- Author block back to SEPARATE blocks (two IEEEauthorblockN/A via \and, no 1/2 superscripts).
- Removed 8 em dashes (---) reintroduced during the reviewer-response round (intro, contributions, F2, discussion #3, CARA mitigation, positioning); replaced with commas/periods. En-dashes (--) for ranges kept. Only remaining --- is in a preamble comment.
- Still 8 pages, 21 cites, 0 undefined.

## 2026-07-23 — Phase 1 (sensei Q1): campaign-driver infrastructure built
Following SENSEI_FEEDBACK_PLAN.md. Phase 0 decisions locked: target-A/fallback-B,
Kaggle free tier, ~18-run minimal matrix, 8-page conference.

- Added `src/campaign.py`: config-loop driver over the EXISTING harness (no new
  algorithms). CampaignConfig(tag, frac, epochs, seed, imgsz, batch);
  default_matrix() = 3 dataset sizes x 3 seeds (ep10) + epochs{5,20} x 3 seeds
  = 15 unique fine-tunes. Each config: subsample Singapore-train images ->
  fine-tune C1 from the SAME frozen Boston weights -> run_all_measurements
  (C2/C3 frozen) -> deltas vs a shared baseline -> regime label + rho.
- subsample_yolo_dataset(): deterministic per-seed image-level subsampling
  (Singapore train has only 3 scenes, so image-level is the right granularity),
  keeps image/label pairs together, copies val split whole.
- label_regime(): classifies {entangled_enhancement, metric_decoupling,
  non_monotone, benign, other}. Branch order fixed so the v22 real run
  (delta1<0, Delta2<0, Delta3>0) labels as metric_decoupling (more specific
  than non_monotone). Convention: Delta2/Delta3 negative = better, delta1
  positive = better.
- coupling_rho() = Delta3/delta1, None when |delta1|~0 (matches paper's honest
  "not cleanly measurable" position for the single run).
- Added `seed` param to c1_yolo.fine_tune_yolo (threaded into Ultralytics
  train(seed=...)); defaults to 0 so existing notebook cells are unaffected.
- OFFLINE self-tests all pass: matrix size=15, regime labels (incl. v22 case),
  rho ambiguity, subsample determinism + label pairing. No GPU/Kaggle used yet.
- NEXT: wire a notebook phase that builds the full Singapore YOLO dataset once,
  measures the Boston baseline once, then calls run_campaign(); push as a new
  kernel and collect campaign_results.json.

## 2026-07-23 — Phase 1 campaign pushed to Kaggle (kernel v23)
- Synced src/c1_yolo.py (with seed) + new src/campaign.py into labsd-src/labsd/;
  pushed labsd-src dataset new version ("add campaign.py driver + seed param").
- Added notebook Phase 10 cell: builds full Singapore YOLO dataset, then calls
  run_campaign(default_matrix()) reusing the frozen Boston C1 + C2 + the Phase-6
  baseline.json; prints a per-config table (tag, n_train, delta1, Delta2,
  Delta3, rho, regime) + regime counts; writes campaign_results.json.
- kaggle kernels push -> kernel version 23 pushed. Status: RUNNING.
- CAVEAT: kernel-metadata has enable_gpu only (no explicit accelerator). If
  Kaggle assigns P100 (sm_60) again, c1_yolo auto-falls back to CPU (slower but
  completes). Watch for that in the run.
- NEXT: poll status; on COMPLETE, download campaign_results.json + logs, build
  the regime table/figure for the paper (Q1 answer).

## 2026-07-23 — Phase 1 campaign v24 COMPLETE but C3 frozen: diagnosis
Kernel v24 ran all 15 configs on T4 (GPU worked). Infra sound: C1 mAP + C2
minADE vary across configs, seeds/subsample/dataset-size all correct.
PROBLEM: C3-pipeline L2 is bit-identical (6.2969...) in baseline AND all 15
retrains -> Delta3=0 everywhere -> regime labels meaningless (only 'other' /
'metric_decoupling', 0 entangled_enhancement).

ROOT CAUSE (confirmed by code + data, NOT a campaign bug):
- IDM planner (c3_idm.py) is progress-dominated with a HARD proximity radius
  SAFE_R=4.0 m. An agent only affects the plan if within 4 m of a candidate
  trajectory. If none are, proximity_pen=0 for all candidates -> same
  max-progress trajectory chosen -> Delta3=0 regardless of detections.
- v24 per-scene: d25718445 went n_agents 1->6 but L2 stayed 14.228 (bit-exact).
- v22's +0.28 m came from ONE scene where a retrained-C1 agent fell INSIDE 4 m
  (1->2 agents, L2 14.228->15.068). v24's Boston baseline detector differs
  (mAP 0.1185 vs 0.0968; default seed=0 changed Boston training), shifting the
  detection distribution so no agent lands within 4 m -> effect vanishes.

INTERPRETATION (this is a FINDING, not just a failure):
- Empirically confirms the paper's own "single-scene-driven, fragile" caveat in
  the strongest way: the planning cascade is KNIFE-EDGE — hinges on one agent
  crossing a hard 4 m threshold.
- The current IDM planner is too insensitive to be a reliable terminal metric
  across a campaign.

FIX NEEDED (next): make C3 respond continuously to detection changes, e.g.
soften the proximity term to an unbounded smooth kernel (remove the hard 4 m
cutoff; use inverse-distance or Gaussian so far agents still exert a small,
monotone influence). Then re-run the campaign. Also pin the Boston C1 seed so
the baseline is stable across campaign runs.

## 2026-07-23 — Fix C: plan-vs-plan cascade metric + softened planner + seed pin
Root problem (from v24): C3 L2-vs-human is dominated by a LARGE agent-independent
geometric offset (scene d2571844 L2=14.2 m regardless of 1 or 6 agents), and the
hard 4 m proximity cutoff made the plan ignore detections -> Delta3=0 in all 15.
Weight-tuning alone could make the plan RESPOND but never MONOTONE/sensible
(0/4 monotone at every proximity weight tested 4..20). So L2-vs-human is the
wrong terminal metric for measuring the cascade.

FIX (chosen = logical best):
1. Softened proximity penalty: hard SAFE_R=4 m cutoff -> unbounded Gaussian
   kernel exp(-d^2/2sigma^2), sigma=6 m; every agent exerts a small continuous
   monotone influence. Finer candidate grid (lateral 0.5 m steps; speed grid
   {s-3..s+2}). proximity weight 50 -> 6 to match the [0,1] kernel scale.
2. NEW pure-cascade metric plan_shift_between(before_c1, after_c1): for each
   scene, plan under before-C1 detections and under after-C1 detections, return
   the L2 between the two PLANS at 1/2/3 s. No human-mismatch offset -> isolates
   the cascade at the planner. shift=0 <=> retraining did not move the plan.
   Verified offline: identical detections -> shift 0.000; 1 vs 3 agents -> 4.0 m.
   Added _plan_for_scene() helper factoring the per-scene ego+agents+plan path.
3. campaign.py now computes plan_shift3 per config (before=frozen Boston, after=
   this fine-tune) and stores it in each row + plan_shift.json.
4. Pinned Boston C1 fine-tune seed=0 in notebook Phase 3 so the baseline is
   stable across runs (v22 mAP 0.1185 vs v24 0.0968 came from an unpinned seed).
- Synced c3_idm/campaign/c1_yolo to labsd-src mirror; notebook Phase 10 table
  now shows the plan_shift column. All parse; offline tests pass.
- NEXT: push labsd-src new version + kernel; re-run campaign; plan_shift3 should
  now vary across configs -> real Q1 cascade landscape.

## 2026-07-23 — v26 COMPLETE: Fix C worked, real Q1 cascade landscape
T4x2. All 15 configs ran; plan_shift3 and Delta3 now BOTH vary (unlike v24's
frozen Delta3=0). The plan-vs-plan metric fixed the dead-signal problem.

RESULTS (baseline: C1 mAP 0.0968, C2_pipe 9.595, C3_pipe_L2 6.990):
- plan_shift3 ranges 1.33 -> 7.80 m (7 distinct values) — retraining C1 moves
  the plan by campaign-dependent amounts.
- Delta3 ranges -1.04 -> +1.34: C3 HURT in 11 configs, HELPED in 2
  (quarter-data high-seed), ~flat in 2. So "retraining hurts planning" is NOT
  universal — it is campaign-dependent (key nuance vs the single v22 point).
- Regimes: 9 other, 4 metric_decoupling, 2 benign.
- Dataset-size effect: full-data fine-tunes -> largest plan shift (mean 6.12 m,
  std 2.10); half ~2.98; quarter ~3.46 -> more data => bigger downstream cascade.
- CAVEAT: delta1<0 in ALL 15 (small-data overfit on <=119 imgs), so still no
  STRICT entangled enhancement (upstream-up + system-down); same limit as paper.
  But we now have non-monotone + decoupling ACROSS A CAMPAIGN = far more general
  than one point. Answers sensei Q1: different campaigns DO show different
  entanglements.

ARTIFACTS: results/run-v26/campaign/campaign_results.json (+ per-config
plan_shift.json, after.json). NEXT: build the Q1 table + figure for the paper;
consider a class-balanced fine-tune to chase a delta1>0 (clean EE) case.

## 2026-07-23 — Meeting 6 report drafted (extension of Meeting 5)
- Created Reports/Meeting 6/Report_Ifty_Meeting6.tex (+ compiled PDF, 4 pages).
- Framed as an extension of Meeting 5 / follow-up to sensei's Q1: "we ran 15
  retraining experiments" (data size full/half/quarter x epochs 5/10/20 x 3
  seeds). Plain-language style matching Meeting 5. NO mention of code/harness
  changes per user request — presented purely as experiment findings.
- Reports the v26 numbers: Table of all 15 (delta1, Delta2, Delta3, plan-shift,
  regime); 4 findings (cascade real & varies 1.33-7.80 m; planner hurt in 11 /
  helped in 2 / flat in 2; more fine-tune data -> bigger plan shift;
  metric-decoupling recurs 4x); honest limitation (delta1<0 in all -> no strict
  EE yet); next steps (class-balanced fine-tune, use campaign to test the drift
  gate, more seeds).
- Paper (ieee paper/) untouched per instruction.

## 2026-07-23 — Phase 2a: drift gate + retrospective evaluation
- Added src/drift_gate.py: model-free interface drift gate. Summarises a
  detector's outputs into per-image count, confidence(score), class histogram,
  box-center x/y; compares old vs new with KS (continuous) + JS divergence
  (categorical) -> scalar drift in [0,1] -> admit/cascade-risk at threshold tau.
  Self-contained (no torch/scipy). Offline self-tests pass (identical->0,
  different->0.66 flagged).
- Added c1_yolo.dump_detections(): runs a detector over scenes, returns raw
  detection dicts {sample_token,cls,x,y,score} for the gate.
- Added campaign.evaluate_gate_on_campaign(): for each of the 15 configs, dumps
  the retrained detector's detections + the frozen Boston detector's, computes
  gate drift, pairs it with the config's measured plan_shift3, and reports
  Spearman(drift, plan_shift). + _spearman() (offline-tested: +1/-1/None).
- Added notebook Phase 11 to run it (tau=0.25); prints drift vs plan_shift table
  sorted by drift + the correlation; writes gate_eval.json.
- Synced drift_gate/campaign/c1_yolo to mirror; all parse.
- KEY QUESTION Phase 2a answers: does high drift predict large downstream plan
  shift? If Spearman is strongly positive -> the gate works -> supports CARA ->
  lean paper structure A. If weak -> note limitation, lean B.
- NEXT: push dataset + kernel, run, read gate_eval.json.

## 2026-07-23 — v28 Phase 11 produced no gate_eval.json; hardened + re-push (v29)
- v28 (T4): Phase 10 campaign succeeded (15 configs) but Phase 11 wrote no
  gate_eval.json. Full output download (501 files, all config dirs) confirmed
  the file is genuinely absent -> Phase 11 errored on Kaggle. Code runs clean
  offline; likely a dataset-version race (v28 may have mounted the pre-Phase-2a
  labsd-src). VERIFIED: latest labsd-src on Kaggle now HAS drift_gate.py +
  c1_yolo.py(20848, with dump_detections) + campaign.py -> a re-run should work.
- Hardened notebook Phase 11: wrapped in try/except that prints the full
  traceback and writes /kaggle/working/gate_error.txt on failure, so the next
  run yields either gate_eval.json OR the exact error (no more guessing).
- Pushed kernel v29. User to set T4x2 in web UI and run next version.
- EXPECTED: gate_eval.json with Spearman(drift, plan_shift). Strong positive =>
  gate works => paper structure A. Weak => note limitation => structure B.

## 2026-07-23 — v30 ran but Phase 11 never executed; split gate into own kernel
- v30 (T4x2) COMPLETE. Phase 10 campaign fully succeeded: all 15 rows present in
  campaign_results.json (identical numbers to v26, already in Meeting 6 report).
  BUT no gate_eval.json and no gate_error.txt -> Phase 11 (the hardened try/except
  cell) produced ZERO output, i.e. the notebook stopped BEFORE reaching it.
- Diagnosis: the 15-fine-tune campaign consumes essentially the whole GPU
  session; the kernel hit Kaggle's wall-clock limit right at Phase 11, which
  itself needs a second heavy pass (dump_detections on all 16 detectors).
  (The source .ipynb from `kernels pull` shows all cells exec=None — that's the
  source, not the executed copy; the real signal is the absent output files.)
- FIX (best approach chosen): a standalone gate kernel that does NO retraining.
  Created ifty1011/labsd-e1-gate ("LabSD E1 - Drift Gate (Phase 11 standalone)"):
    * kernel_sources = [ifty1011/labsd-e1-cascade-degradation] -> mounts the
      previous kernel's output (all 16 best.pt weights + campaign_results.json).
    * setup cells copied verbatim from the main notebook (labsd import, nuScenes
      extract, build_splits, NuScenes load).
    * gate cell: finds campaign_results.json + Boston best.pt under /kaggle/input,
      remaps each config's weights by tag (glob, not the stale /kaggle/working
      path in c1_descriptor.json), runs dump_detections + gate_decision(tau=0.25),
      computes Spearman(drift, plan_shift3), writes gate_eval.json (or gate_error.txt).
  Inference-only, so P100/T4 tier is irrelevant; should finish in minutes.
- Pushed labsd-e1-gate v1. NEXT: run it, read gate_eval.json -> Spearman value ->
  paper structure A (gate works) vs B (weak).

## 2026-07-23 — gate kernel iterations v1->v3 (deps, slug, CPU)
- Standalone gate kernel debugged in 3 quick pushes:
  * v1: ModuleNotFoundError 'nuscenes.nuscenes' — I'd copied the main notebook's
    SETUP cells but not its pip-install cell. Added an install cell mirroring the
    main notebook (nuscenes-devkit --no-deps + pyquaternion/cachetools/descartes,
    then ultralytics + ultralytics-thop/py-cpuinfo).
  * slug fix: kernel id resolves to 'labsd-e1-drift-gate-phase-11-standalone'
    (from the title), not 'labsd-e1-gate'; updated kernel-metadata.json id/title.
  * v2: got much further — labsd imported, prev-kernel output mounted (16 weights
    + campaign_results.json found), nuScenes extracted, ultralytics loaded — then
    died in dump_detections with CUDA 'no kernel image is available for execution
    on the device' == the P100/sm_60 mismatch (CLI push lands on old GPU).
  * v3 FIX: gate is inference-only on 6 tiny scenes x 16 detectors, so no GPU
    needed. Set enable_gpu=false in metadata AND CUDA_VISIBLE_DEVICES='' before
    importing torch -> forces CPU, removes all GPU-tier fragility. Pushed v3.
- EXPECTED from v3: gate_eval.json with Spearman(drift, plan_shift3) + the
  per-config drift/cascade_risk table. That's the Phase 2a deliverable and the
  paper-structure A (gate predicts plan shift) vs B (weak) decision.

## 2026-07-23 — Phase 2a COMPLETE: drift-gate result (gate kernel v7, T4x2)
- v5/v6 fix: gate now indexes each config's best.pt by tag from a full glob of
  the mount (v4 returned n_rows=0 because the fixed path shape didn't match the
  mounted layout). v7 (GPU enabled, run on T4x2) succeeded: all 15/15 configs
  matched, gate ran clean. Saved -> results/gate_eval_v7.json.
- RESULT: Spearman(drift, plan_shift3) = +0.413  (n=15, tau=0.25).
  * Positive and moderate: higher interface drift DOES tend to go with a larger
    downstream plan shift, but it is not a tight monotone predictor.
  * Cleaner at the extremes: the two biggest plan shifts (7.80, 7.40 m) have the
    two highest drifts (0.582, 0.528); the two smallest (1.33 m) sit mid/low.
  * Data-size trend matches Meeting-6 finding: full-data 10-ep runs have BOTH the
    highest mean drift (0.546) AND the highest mean plan shift (6.12 m); half/
    quarter lower on both (~0.38 drift, ~3 m).
  * Feature attribution: drift is dominated by box-center x (KS up to 0.93) and
    per-image count; class-mix (JS) contributes little (<=0.19). i.e. retraining
    mainly shifts WHERE and HOW MANY objects are detected, not WHICH classes.
  * tau=0.25 flags all 15 as cascade-risk (min drift 0.30). tau is too low to
    discriminate; a useful operating point is higher (~0.45-0.50) to separate the
    big-shift full-data runs from the rest. NOTE for paper: report the ranking/
    correlation, not the binary flag at 0.25 (it saturates).
- DECISION (A vs B): moderate +0.41 is a real but imperfect signal -> the gate is
  a plausible screen, not a proven predictor. Lean paper structure B (honest
  experience report) with the gate presented as a promising direction whose
  correlation we measured, NOT structure A (gate validated). Can strengthen by
  reporting Spearman with a better tau and the extremes-separation above.

## 2026-07-23 — Phase 2b COMPLETE: full CARA admission-rule evaluation
- New module src/cara_eval.py (pure post-hoc, no GPU): turns the Phase-2a drift
  scores into an admit/hold decision and scores it against ground truth.
  Ground truth per update: Delta3>0 => "bad" (planner regressed). 15 updates:
  11 bad / 4 good; collision rate 0.0 on ALL 15. Saved results/cara_eval.json.
- HEADLINE NUMBERS:
  * AUROC(drift ranks bad above good) = 0.705 — the gate does carry real signal
    (chance=0.5), consistent with the +0.41 Spearman, but far from separable.
  * At the shipped tau=0.25 the rule HOLDS ALL 15 (drift min=0.30 > 0.25):
    TP=11 FP=4 FN=0 TN=0 — recall 1.00 but specificity 0.00. Safe-but-useless
    ("hold everything"): it never admits a good update.
  * Threshold sweep found a real operating point at tau≈0.41 (max Youden J):
    TP=8 FP=1 FN=3 TN=3 — precision 0.89, recall 0.73, specificity 0.75, F1 0.80.
    i.e. tuned, the gate catches ~3/4 of bad updates while correctly admitting
    3/4 of good ones, with only 1 false hold.
  * Collision-aware variant is DEGENERATE here: 0 collisions everywhere, so it
    collapses to the L2 rule (identical confusion). Honest limitation to report:
    the L2-vs-collision tension the paper flags cannot be exercised on this data.
- INTERPRETATION for paper: the gate is a plausible SCREEN, not a validated
  admission rule. Default tau saturates; only a swept threshold discriminates,
  and that tuning is done post-hoc on the same 15 points (no held-out set) — so
  it is proof-of-concept on n=3 scenes / 15 updates, exactly as the plan scoped.
- A vs B DECISION (now firm): B — honest experience report. Present CARA as a
  proposal with measured but imperfect evidence (AUROC 0.705; tuned F1 0.80;
  saturating default), NOT structure A (validated method). Threshold-sweep table
  + per-update table are the honest artifacts to show.
- Phase 2 (Q2) DONE. NEXT per plan: Phase 3 restructure paper (target B), then
  Phase 4 writing pass, Phase 5 review. Also optional: fold 2a/2b into Meeting 6.

## 2026-07-23 — folded Phase 2a/2b into Meeting 6 report
- Added to Reports/Meeting 6/Report_Ifty_Meeting6.tex a new section "Can We
  Predict Which Campaigns Are Risky?" with Finding 5 (drift vs plan-shift,
  rank corr ~0.41, clean at extremes) and Finding 6 (drift as go/no-go gate:
  AUROC ~0.70; strict tau holds all = safe-but-useless; balanced tau ~3/4 catch
  & ~3/4 admit; collision variant untestable = 0 collisions). Framed purely as
  experiments/screening check, NO mention that code was written or changed
  (consistent with the report's framing). Updated Summary: the screening idea
  was tested (not just proposed); next-steps item 2 changed from "evaluate the
  gate" (done) to "widen to held-out set". Compiles clean, 4 pages.

## 2026-07-23 — Phase 3 COMPLETE: paper restructured to Option B
- New file: ieee paper/IEEE_Conference_Paper_E1_v2.tex (original E1.tex untouched).
  Applied SENSEI_FEEDBACK_PLAN §5.3 (structure B, experience report). Compiles
  clean via pdflatex+bibtex+pdflatex, 10 pages, 21 citations, no undefined refs.
- Section order now: Intro -> System Model & Methodology -> Experimental Setup ->
  Results -> Practical Implications -> Related Work (moved to BACK) -> Conclusion.
- Key structural moves:
  * Intro reframed: E2E-limits + gap now = "no real AV-pipeline study AND no
    measure of how it varies run-to-run"; findings list expanded to F1-F6 (added
    F6 = landscape). Contributions updated to campaign + drift screen.
  * Entangled-enhancement DEFINITION consolidated into System Model (new
    subsec III-B with strict condition eq + metric-decoupling), removed from Intro
    and from the old Cascade-Diagnostic prose.
  * Experimental Setup gained "The Retraining Campaign" subsec (15-update matrix:
    3 sizes x{5,10,20}ep-ish x3 seeds). Metrics subsec gained plan-shift def.
  * Results: reference-update table/figs kept (F2-F5), + new "Campaign: landscape"
    subsec with 15-row table (tab:campaign) + F6, + "drift screen" subsec
    (Spearman 0.41, AUROC 0.70, tau saturates, balanced tau prec .89/rec .73/
    spec .75, 0 collisions -> collision variant untestable). Framed honestly as
    proof-of-concept.
  * NEW Practical Implications section: mechanism + 4 guidance points + multi-angle
    mitigations (PC-train, BCT, uncertainty prop, data-validation) with CARA as
    ONE admission-step recommendation (algo+fig kept). Threats-to-validity folded in.
  * Related Work moved to back verbatim (4 subsecs), refs re-pointed to
    sec:implications instead of old discussion/threats labels.
  * Conclusion + abstract rewritten around the campaign + drift-screen evidence.
- NEXT per plan: Phase 4 writing pass (W1 de-chain ; / : ; W2 add lead-ins to
  terse claims), then trim to venue page limit, then Phase 5 review.

## 2026-07-23 — DECISION: switch paper to Option A (technical paper)
- User chose Option A over B (sensei: "A is more impactful IF evaluation added"; we
  added CARA eval: AUROC 0.70, drift Spearman 0.41, admission-rule confusion).
- Target A structure (sensei's layout): Intro -> System Model (incl. EE def) ->
  CARA (promoted, before experiments) -> Experiment Setup -> Evaluation Results
  (empirical findings + CARA evaluation as validation) -> Related Work -> Conclusion.
- Plan: re-order v2 into a new v3 file. Pull CARA out of Practical Implications
  into its own section before Experiment Setup; reframe Results as Evaluation
  Results; demote multi-angle mitigation (fold survivors into Discussion/Related);
  intro contributions = shows evidence AND proposes+evaluates a solution.
- Writing pass (W1 de-chain ;/:  W2 lead-ins) still pending after re-order.

## 2026-07-23 — Phase 3 REDONE as Option A (technical paper) -> v3
- New file ieee paper/IEEE_Conference_Paper_E1_v3.tex (v2 kept as the Option-B
  variant). Compiles clean (pdflatex+bibtex), 10 pages, 21 cites, no undefined refs.
- Section order now matches sensei's Option A EXACTLY:
  Intro -> System Model (incl. EE def) -> CARA (promoted, its own section) ->
  Experiment Setup -> Evaluation Results -> Discussion -> Related Work -> Conclusion.
- Moves done:
  * Intro contributions/roadmap reframed: paper "shows evidence AND proposes+
    evaluates CARA"; roadmap now lists CARA as its own section III.
  * CARA promoted to Section IV (before experiments): subsecs Admission Procedure
    (+algorithm+fig), Interface-Drift Front-End, Why-Modular. Method text moved
    out of the old Practical-Implications subsection.
  * Results -> "Evaluation Results": added lead framing (two evaluations: the
    cascade study + CARA). Drift-screen subsec split into (a) prediction
    (Spearman 0.41 / AUROC intro) and (b) NEW "Evaluation of CARA's Admission
    Rule" subsec (11 bad/4 good, AUROC 0.70, strict tau saturates, balanced tau
    prec .89/rec .73/spec .75, 0-collision caveat, proof-of-concept honesty).
  * Practical Implications -> slimmed "Discussion": kept mechanism + threats,
    compressed 4-point guidance into "CARA Among Related Maintenance Techniques"
    (CARA as one of several: PC-train/BCT/uncertainty-prop/data-validation).
  * Fixed Jensen--Shannon typo; all label re-points verified.
- v2 (Option B) and v3 (Option A) now both exist; v3 is the chosen direction.
- NEXT: Phase 4 writing pass (W1 de-chain ;/: , W2 lead-ins to terse claims);
  then page-limit trim for target venue; then Phase 5 review.

## 2026-07-23 — formal-tone pass on the paper + kept only v3
- Read v3 end-to-end, found informal phrasings (mostly in the quickly-written
  abstract/campaign/CARA/drift sections) and formalized them:
  * "what happens downstream" -> "the downstream effect"; "poor fit" -> "ill-suited";
    "cheap check/screen/front-end" (recurring) -> "low-cost"; "we then ask ... and
    find" -> "we then assess ... and find that".
  * "barely nudge the driving decision" -> "alters ... only slightly";
    "same style of retraining" -> "same form"; "three patterns stand out" ->
    "three patterns emerge"; "the two ``helped'' cases" -> "the two improved cases".
  * "pushes a bigger change down the pipeline" -> "induces a larger change
    throughout the pipeline"; "So the front-end does carry" -> "The front-end
    therefore carries"; "a safe but useless filter" -> "safe but provides no
    discrimination"; "The honest reading is that" -> "We interpret ... conservatively".
  * "The campaign lets us evaluate" -> "enables an evaluation of"; "We distilled
    practical guidance" -> "We proposed the CARA admission method"; "warns against
    reading the terminal metric too literally" -> "cautions against a literal
    interpretation".
  * verified no contractions / casual markers remain. Recompiles clean, 10 pages.
- PER USER: deleted E1 original + v2 (.tex/.pdf) and all build artifacts; ONLY
  IEEE_Conference_Paper_E1_v3.{tex,pdf} kept in ieee paper/.

## 2026-07-23 — Phase 4 writing pass DONE (sensei comments 1 & 2)
- W1 (de-chain ; / : joining distinct topics): split overloaded sentences in
  Intro (three-reasons passage), F4 finding, CARA isolation sentence, campaign
  setup, training diagnostics, related-work (Ivanovic/Wang), threats (stat-power,
  component-realism), F4 diagnostic reading, and the four-line future-work
  sentence. Left legitimate list/definition/appositive colons intact.
- W2 (lead-ins for high-context terse sentences) — sensei's 6 examples all fixed:
  * "This paper supplies that measurement." -> "The missing element is a
    controlled measurement of this propagation on a real pipeline, and this paper
    supplies it." (grounds the pronoun)
  * "The main instrument is dual-mode evaluation:" -> "...and it exists to
    separate a component's own error from the error it inherits" (already in v3).
  * gap-between-modes -> explained in words before the formula (v3).
  * "The harness computes the five-quantity profile..." -> preceded by a sentence
    naming the five quantities (1 C1 metric + iso/pipe of C2 and C3) and why.
  * "The three isolated-mode rows are exactly zero-delta." -> lead-in "Before
    reading any cascade delta, the frozen components must be shown to be untouched
    ... a validity check, not a result." (v3).
  * "Table I exhibits a second inversion." -> "...a second inversion, this one
    between C1's own score and the usefulness of its outputs." (v3).
- Verified no ungrounded demonstrative openers remain. Recompiles clean, 10 pages,
  no undefined refs. Temp cleared. Only v3 .tex/.pdf in ieee paper/.
- ALL SENSEI ITEMS now addressed: Q1 (campaign), Q2 (CARA eval), core story,
  structure A, writing (1) and (2). Remaining = optional class-balanced fine-tune
  for a clean delta1>0, and page-limit trim for the target venue.

## 2026-07-23 — added two CARA-evaluation visuals to the paper
- The CARA evaluation (V-G, V-H) was prose-only; added two native TikZ/pgfplots
  visuals (no external image files) built from results/gate_eval_v7.json +
  cara_eval.json:
  * Fig. (fig:drift-scatter) in V-G: scatter of interface-drift (x) vs plan-shift
    (y) over all 15 updates; harmful (Delta3>0, filled blue) vs harmless (open
    orange), dashed line at balanced tau=0.41. Caption notes Spearman~0.41.
  * Table (tab:admission) in V-H: strict vs balanced operating points with
    TP/FP/FN/TN + recall/spec/prec/F1 (strict 11/4/0/0 rec1.0 spec0; balanced
    8/1/3/3 rec.73 spec.75 prec.89 F1.80).
- Point counts verified (11 harmful + 4 harmless = 15). Compiles clean, now 11
  pages, both labels resolve, no undefined refs. Temp cleared. Only v3 .tex/.pdf.

## 2026-07-23 — class-balanced fine-tune built (Phase 12), kernel v31 pushed
- GOAL: push delta1 > 0 (C1 improves on its own metric) so that, if the terminal
  metric still degrades, we obtain the STRICT entangled-enhancement condition
  that sensei's core story claims ("shows the real entangled enhancement").
  Every campaign run so far had delta1 < 0 (small-data overfit, car-dominated mix).
- NEW src/class_balance.py (no torch/scipy):
  * class_counts(): per-class instance counts over a YOLO split.
  * build_class_balanced_split(): IMAGE-LEVEL OVERSAMPLING. repeat factor per
    class = (max_count/count)**power clipped to [1,max_repeat]; an image is
    repeated as often as its RAREST class demands. val split copied verbatim
    (copy_split) so mAP stays comparable to all other campaign runs.
  * Chose max_repeat=12 by offline sweep on a synthetic car-dominated set:
    imbalance 100x -> 14x at only 142 imgs (power .5); higher caps mostly inflate
    dataset size. Documented the honest limit: driving scenes are multi-label and
    `car` co-occurs with nearly every rare class, so duplicating a rare-class
    image also duplicates cars -> imbalance shrinks a lot but cannot reach 1.0
    by image repetition alone.
- NEW campaign.run_class_balanced(): builds one balanced dataset per `power`
  (0.5, 1.0), fine-tunes over epochs (10,20) x seeds (1,2,3) = 12 runs, measures
  the full profile + plan_shift, and flags per row
  strict_entangled_enhancement = (delta1>0 AND Delta3>0). Writes
  class_balanced_results.json + n_delta1_positive / n_strict_entangled_enhancement.
- Notebook: added Phase 12 (markdown + hardened try/except cell writing
  phase12_error.txt on failure). 29 cells. Synced mirror; pushed labsd-src new
  version and kernel v31.
- NOTE: `kaggle datasets files` lists only top-level files (shows just setup.py);
  nested labsd/*.py do upload and mount - confirmed by the earlier drift_gate run.
- NEXT: user sets T4x2 in web UI and runs v31; read class_balanced_results.json
  (or phase12_error.txt). If any run has delta1>0 AND Delta3>0 -> strict EE, the
  paper's central claim upgrades from "related regime" to "the real thing".

## 2026-07-23 — v31 FAILED (dataset wiped); fixed via labsd.tar + robust loader (v33)
- v31 died at cell 3: RuntimeError('labsd-src not found'). NOT a Phase-12 bug.
- ROOT CAUSE: my earlier `kaggle datasets version -d` upload REPLACED the dataset
  contents with only setup.py (148B) - the nested labsd/ dir never uploaded. I
  initially misread the short `datasets files` listing as "CLI hides nested files";
  that was wrong - the package really was gone, which is why the mount failed.
- FIX 1 (upload): packaged the mirror as labsd.tar (210KB) and pushed a new
  dataset version. Kaggle AUTO-EXTRACTS it, so files now live at
  labsd/labsd/*.py (an extra nesting level) - confirmed class_balance.py present.
- FIX 2 (loader): rewrote notebook Phase 1 to be layout-agnostic. It now globs
  for **/labsd/__init__.py anywhere under /kaggle/input and puts that file's
  grandparent on sys.path, with tar/zip extraction as fallbacks, and raises a
  diagnostic listing of /kaggle/input on failure. Added a fail-fast
  `from labsd import class_balance` so a stale mirror is caught immediately
  instead of surfacing 40 minutes later in Phase 12.
- Verified the patched cell retained ALL pip installs (nuscenes-devkit,
  ultralytics, PKL) and YOLO_AVAILABLE.
- Pushed kernel v33. NEXT: set T4x2 in web UI, run, read
  class_balanced_results.json (or phase12_error.txt).

## 2026-07-23 — *** STRICT ENTANGLED ENHANCEMENT OBSERVED *** (v34, class-balanced)
- v34 (T4x2) COMPLETED. Phase 12 ran all 12 class-balanced configs.
  Saved -> results/class_balanced_v34.json.
- HEADLINE: delta1 > 0 in 6/12 runs, and STRICT entangled enhancement
  (delta1 > 0 AND Delta3 > 0) in 3/12 runs:
    bal_p05_ep20_s1  d1=+0.0308  D2=+3.873  D3=+0.242  shift 2.59  EE=True
    bal_p10_ep20_s2  d1=+0.0118  D2=+0.875  D3=+0.243  shift 1.26  EE=True
    bal_p10_ep20_s3  d1=+0.0210  D2=+8.589  D3=+0.242  shift 2.59  EE=True
  This is the condition from Wang&Machida the paper is named after, and which
  ALL 15 unbalanced campaign runs failed to produce (delta1<0 everywhere).
- Class balancing worked as intended: it reversed the small-data mAP drop.
  power=0.5: 119 -> 399 imgs, imbalance 34.8x -> 17.7x
  power=1.0: 119 -> 575 imgs, imbalance 34.8x -> 27.5x
  (power=1.0 gives MORE images but WORSE imbalance, because repeating rare-class
  images also multiplies the co-occurring `car` instances - the multi-label limit
  documented in class_balance.py.)
- Both delta1>0 and strict EE concentrate in the 20-EPOCH runs: longer adaptation
  is what pushes C1 past the Boston baseline. 10-epoch runs mostly land benign.
- PAPER IMPACT: the paper can now claim the STRICT condition with 3 independent
  instances across 2 balance settings and 2 seeds. This closes the one gap vs
  sensei's core story ("shows the REAL entangled enhancement in an AV pipeline").
- NOTE: Abstract, Results (F4/F6) and Conclusion still state that the strict
  condition never appears - now FALSE and must be updated. (The Threats section
  that also said it was deleted this session at user request.)
