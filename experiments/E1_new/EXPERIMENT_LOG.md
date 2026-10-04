# E1 (new stage) — Experiment Log

Next stage of experiment E1: all three modules are ML components, 350-scene
subset of nuScenes, hypotheses H1 (existence) and H2 (symptom) of the Meeting 6
report. The earlier experiments and their log are in experiments/E1_kaggle_old/.

Every action, decision, blocker, fix and result is appended here.

---

## 2026-09-23 — Next stage started: hypothesis clarified, 350-scene subset kernel pushed

Sensei set three tasks at the last meeting: (1) state the hypothesis clearly
before experimenting, (2) reduce the data, (3) replace the deterministic C2 and
C3 with machine learning models. Meeting 6 report written in
Reports/Meeting 6/Report_Ifty_Meeting6.tex.

HYPOTHESIS (H1): with learned C2/C3 trained on the OLD C1's output, retraining
C1 alone gives delta1>0 and Delta3>0. Tested with three ways of training C2/C3:
isolated (ground truth), old-pipeline (old C1 output), new-pipeline (new C1
output). H2: the fraction of updates with EE under old-pipeline training
estimates 1-c1 of Wang & Machida.

MODELS (all learned, no rules): C2 AutoBot-Ego (BSD-3), fallback Social-LSTM
(write ourselves, no license); C3 PlanT (MIT), fallback AD-MLP with object
input (write ourselves, no license). C3 gets no ego speed input.

DATASET (Task 2): 350 scenes, full-size CAM_FRONT keyframes, no shrinking.
  boston_c1        100  (even stride over boston_train 280)   -> train old C1
  boston_c23       100  (even stride over boston_val 187)     -> train C2/C3
  singapore_update 100  (even stride over singapore_train 229)-> update C1
  singapore_test    50  (stride 154/50 over singapore_val)    -> SAME 50 test
                                                                 scenes as paper
Split logic reproduces labsd.splits.partition_by_location + cap_val_scenes.
Estimated ~3-4 GB (vs 45 GB). Kernel builds it from the public dataset.

KERNEL: ifty1011/labsd-e1-subset350 (experiments/E1_kaggle/subset_350/),
CPU only, no internet, input sahangunasekara92/nuscenes-v1-0-full-keyframes.
Output /kaggle/working/nuscenes_subset350/ (v1.0-trainval metadata, selected
CAM_FRONT images, splits_350.json, subset_summary.json). Pushed v1, RUNNING.
NEXT: when complete, read subset_summary.json for real image count and GB;
attach the kernel output as input (kernel_sources) in the next kernels.

## 2026-09-23 — 350-scene subset COMPLETE (kernel v1, 434.5 s)
subset_summary.json (from the user, Kaggle API was rate-limited):
  scenes: boston_c1 100, boston_c23 100, singapore_update 100, singapore_test 50
  CAM_FRONT keyframes: 4008 / 4046 / 3997 / 2020 = 14,071 total, 0 missing
  images 2.043 GB + metadata 2.602 GB = 4.645 GB total (vs 45 GB)
singapore_test = 2020 images, the same count as the paper's 50-scene test set,
which confirms the test scenes are identical to the paper.
Metadata is the FULL v1.0-trainval (all 850 scenes) and is larger than the
images; it can be trimmed to the 350 scenes if size matters.
Output usable via kernel_sources: ifty1011/labsd-e1-subset350.

## 2026-10-03 — Stage 0 pushed: old C1 with two seeds + one update each (350-scene subset)
First run of the next stage (Meeting 6 report, Sec. 4.1 "first run").
KERNEL: ifty1011/labsd-e1-stage0-c1 v1 (experiments/E1_kaggle/stage0_c1/),
  GPU T4 (--accelerator NvidiaTeslaT4; notebook asserts sm>=70, no CPU fallback),
  inputs: dataset ifty1011/labsd-src, kernel_sources ifty1011/labsd-e1-subset350.
STEPS:
  1. old C1 = YOLO11n from yolo11n.pt on boston_c1 (100 scenes), seeds 0 and 1
  2. new C1 = fine-tune each old C1 on singapore_update (100 scenes), same seed
  3. evaluate all 4 on singapore_test (50 scenes, same as paper): mAP50, mAP50-95, P, R
  4. dump detections of all 4 on singapore_test (input to the H2 label-free measures)
CONFIG: epochs 10, imgsz 640, batch 8 (same as earlier experiments),
  ultralytics pinned to 8.4.105 (the version of the earlier working runs).
CHANGE vs earlier experiments: checkpoint = last.pt and training uses val=False,
  so the test scenes are never used to select a checkpoint (earlier runs picked
  best.pt on singapore_val, a known gap).
MODEL SAVING (new standing rule, user request): every model is saved to
  /kaggle/working/models/<name>.pt with manifest.json (config, seed, init,
  sha256, test metrics). Download to experiments/E1_kaggle/models/stage0/ and
  reuse via kernel_sources; never retrain. models/ and *.pt are gitignored.
QUESTIONS IT ANSWERS: seed gap of old C1 (random variation) vs delta1 of an
  update; old C1 mAP50 on 100 scenes vs paper 0.137 on 280 scenes.
EXPECTED TIME: ~4 trainings x ~10 min (4k images, 10 ep; paper: 11k images 27 min)
  + evals + detection dumps, ~1-1.5 h. STATUS: RUNNING.

## 2026-10-03 — 350-scene subset rebuilt on account 2 (for parallel runs)
Account 2 cannot read account 1's private kernel output, so the same subset
notebook was pushed as rai73416/labsd-e1-subset350 v1 (experiments/E1_kaggle/
subset_350_acct2/, CPU only, no GPU quota). Same split logic and the same
seed-free stride selection -> identical 350 scenes. Pushed with the isolated-
HOME method (acct2 token copied into a temp HOME). PLAN: run P1/P2 on acct1
and P3/P4 on acct2 once Stage 0 is done; Stage 0 models will be uploaded to
acct2 as a private dataset. Account 2 has its own rai73416/labsd-src.
STATUS: RUNNING.
2026-10-03: the first acct2 subset run was cancelled by accident by the user
(empty log, CANCEL_ACKNOWLEDGED). Re-pushed rai73416/labsd-e1-subset350; RUNNING.

## 2026-10-03 — Stage 0 v1 ERROR (45 s, no training), fixed and re-pushed as v2
ERROR: NuScenes devkit asserts that every map image exists at start-up
  (map mask .../nuscenes_subset350/maps/53992ee3....png does not exist). The
  350-scene subset copies metadata + CAM_FRONT images only, no maps/.
FIX (no subset rebuild): build /kaggle/working/nusc_root with symlinks to the
  subset's v1.0-trainval/*.json and samples/, and a rewritten map.json whose
  records point to a 1x1 placeholder PNG. An empty map.json was tried first in
  a local test and FAILS (devkit indexes map[0] in __make_reverse_index__); the
  placeholder version was tested locally with the real devkit and loads.
  Maps are not used by C1 training, evaluation or detection dumps.
  Same fix is needed in every later kernel that loads the subset (both accounts).
Pushed ifty1011/labsd-e1-stage0-c1 v2 (T4). STATUS: RUNNING.

## 2026-10-03 — Folders reorganised (user request)
experiments/E1_kaggle -> experiments/E1_kaggle_old (earlier experiments, paper data).
New-stage work moved to experiments/E1_new/: subset_350/, subset_350_acct2/,
stage0_c1/, analysis/, models/, results/{stage0_v1,stage0_v2,subset350,subset350_acct2}.
This log was split off from the old one at the 2026-09-23 entry; the old log
ends with a pointer here. labsd source stays in E1_kaggle_old/src (Kaggle
dataset ifty1011/labsd-src is unchanged). Memory rules updated to the new paths.

## 2026-10-03 — Stage 0 COMPLETE (v2, 2546 s = 42 min, T4)
Account 2 subset: rai73416/labsd-e1-subset350 COMPLETE (summary download hit
Kaggle 429 rate limit; retrying).
TIMING: deps+load 103 s | build YOLO sets 123 s | old C1 579/535 s |
  new C1 493/487 s | 4 evals ~11 s each | 4 detection dumps ~45 s each.
DATA: boston_c1 3923 train images with labels, singapore_update 3575,
  singapore_test 1805 images with labels (of 2020; images with no labelled
  object are dropped by build_yolo_dataset).
C1 ON singapore_test (checkpoint last.pt, no test-set selection):
  model       mAP50   mAP50-95  P      R      detections
  c1_old_s0   0.1086  0.0483    0.363  0.105  3563
  c1_new_s0   0.1836  0.0833    0.341  0.194  3603
  c1_old_s1   0.1173  0.0543    0.383  0.116  3705
  c1_new_s1   0.1850  0.0855    0.364  0.185  3409
  delta1 (mAP50): s0 +0.0750, s1 +0.0677
  seed gap of old C1 (mAP50): 0.0087  -> delta1 is ~8x the seed gap
  paper's old C1 mAP50 (280 scenes, best.pt picked on the test set): 0.137;
  100 scenes give 0.109-0.117. Lower, as expected from fewer scenes and no
  test-set checkpoint selection.
VERDICT on "is 350 scenes enough" (C1 part): yes. The update effect on C1 is
  large and clearly above random variation.
MODELS saved locally: E1_new/models/stage0/{c1_old_s0,c1_new_s0,c1_old_s1,
  c1_new_s1}.pt + manifest.json (sha256 verified after download) + detections/.
NOTE: Ultralytics downloaded yolo26n.pt during its AMP self-check; not used
  for training (base weights are yolo11n.pt, confirmed in manifest).

H2 FIRST LOOK, C1 output only (analysis/h2_c1_measures.py, no ground truth,
1335 images with any detection, objects matched within 2 m and same class):
  measure                    seed noise old  seed noise new  update s0  update s1
  objects/image JS           0.0065          0.0052          0.0068     0.0140
  type mix JS                0.0056          0.0027          0.0372     0.0367
  confidence Wasserstein     0.0070          0.0036          0.0356     0.0350
  confidence KS              0.032           0.030           0.091      0.075
  forward position W (m)     9.26            4.15            12.71      11.02
  lateral position W (m)     1.49            5.90            2.55       5.26
  only-one-version fraction  0.399           0.368           0.509      0.510
READING: type mix, confidence and only-one-version fraction separate an update
  from seed noise clearly (5-10x for type mix and confidence). Objects per
  image and the position distances do NOT: seed noise is as large as the
  update. Position measures look noisy (ground-plane back-projection of 2D
  boxes is far-range sensitive). Whether these changes predict HARM (H2) needs
  C2/C3 and many updates; this only shows which measures are above noise.

## 2026-10-03 — Pushed (ccfb7e6); Stage 1 started: ML C2/C3 (plan approved)
Plan: /Users/rashedul/.claude/plans/toasty-honking-snowglobe.md (user approved).
User chose to train C2/C3 LOCALLY on the Mac (Apple M3, 16 GB, MPS); Kaggle only
extracts images -> detections + ground truth once.
STEP 1 (Kaggle): ifty1011/labsd-e1-stage1-data v1 pushed (T4). Inputs: labsd-src,
  kernel_sources subset350 + stage0-c1 (Stage 0 models reused, sha256 verified in
  the kernel, never retrained). Exports per keyframe for boston_c23 and
  singapore_test: CAM_FRONT ego pose, GT objects (instance, class, global x/y/yaw,
  size, visible-in-CAM_FRONT, lidar+radar points), detections of the 4 C1 models.
  Consistency check: singapore_test detection counts must equal Stage 0's.
STEP 2 (local): venv now has torch 2.14.1 (MPS available), numpy 2.4.4, scipy 1.18.1.
  New package experiments/E1_new/mlpipe/:
  data.py      samples at EVERY keyframe with 3 s of future (old code planned only
               at the first keyframe and its "GT" velocity used future frames).
               C2 input = objects of the last 4 keyframes as an unordered set (no
               track IDs, no hand-written tracker: association is learned by
               attention). C2 target via one-to-one Hungarian assignment (same
               class, radius max(2 m, 0.15 x range)) - labels/metrics only.
               C3 target = human ego path 0.5..3 s; command L/S/R (VAD rule, +-2 m).
  models.py    C2 AutoBot-Ego style (d128, 2+2 layers, 6 modes, bivariate Gaussian,
               WTA NLL + mode CE + ADE/FDE). C3 PlanT style (object tokens + command
               + CLS, 4 layers, GRU 6 waypoints, L1). No ego speed/history in C3.
  train.py, evaluate.py (minADE6, L2@1/2/3s point convention, oriented-box
               collision vs GT future boxes, object-use check, H2 C2 measures).
  selftest.py  synthetic data: transforms (parked car stays still), ego path,
               command, history, assignment, C2/C3 overfit, determinism.
               RESULT: ALL PASSED (C2 1.26 m vs stay-still 9.19 m; C3 L2@3s 1.29 m
               vs mean path 4.50 m; evaluation bit-identical on repeat).
  run_stage1.py  three training settings (isolated / old-pipeline /
               new-pipeline) for P1, evaluation with old/new C1 and the other
               old seed (random variation); models saved to models/stage1 with
               manifest and reused if present.
STATUS: waiting for the Stage 1 data kernel.

## 2026-10-03 — Stage 1 data COMPLETE; smoke run OK; full Stage 1 started locally
Kernel ifty1011/labsd-e1-stage1-data v1: COMPLETE in 953 s (T4).
  boston_c23: 100 scenes, 4046 keyframes; singapore_test: 50 scenes, 2020 keyframes.
  Detections on boston_c23: old_s0 25,286 | new_s0 20,511 | old_s1 26,813 | new_s1 23,663
  (the Singapore-updated C1 finds ~15-20% fewer objects on Boston images).
  singapore_test counts identical to Stage 0 (3563/3603/3705/3409): match=true.
  Files: E1_new/data/stage1/{boston_c23,singapore_test}.json (15.0 MB, 3.5 MB).
FIX: torch.use_deterministic_algorithms(True) breaks MPS training
  (index_put_with_accumulate_mps). Now OFF for training (MPS), ON for every
  evaluation call (CPU). Isolation check still exact.
SMOKE (5 train scenes, 3 test scenes, 2 epochs): end-to-end OK, all 3 settings
  trained + saved, isolation_exact=true. Smoke outputs deleted.
FULL RUN: python -m mlpipe.run_stage1 --c1-seed 0, then --c1-seed 1 (C2 40 epochs,
  C3 60 epochs, seed 0), local MPS. STATUS: RUNNING.

## 2026-10-03 — Stage 1 full run COMPLETE (local M3, 22:10-22:47, both C1 seeds)
Training per setting 5-8 min (C2 40 ep + C3 60 ep, 3445 Boston anchors). 12 models
saved in models/stage1 (C2 718,495 params, C3 619,010 params) + manifest.
Results: results/stage1/p1_c1s{0,1}_seed0.json, log results/stage1_run.out.
1720 test anchors (every keyframe with 3 s future), isolation_exact=true everywhere,
object use (plan change at 3 s when objects removed) 3.5-8.6 m: C3 uses objects.

H1 numbers (old C1 -> new C1, test set):
  setting       seed  delta2 minADE  delta3 L2@3s | seed-noise delta2  delta3
  isolated      0     -0.161         -0.281       | +0.118             -0.477
  old-pipeline  0     -0.307         -0.100       | -0.019             -0.253
  isolated      1     -0.222         +0.266       | -0.135             +0.475
  old-pipeline  1     -0.096         -0.093       | +0.063             +0.081
  new-pipeline vs old-pipeline (each with its own C1): L2@3s -1.49 (s0), -0.70 (s1)
  No entangled enhancement in either update; all deltas are within seed noise.

SANITY BASELINES (required by the plan) - FAILED ON THE TEST SET:
  singapore_test: C2 "stay still" ADE 4.48 m; isolated C2 minADE 4.68/4.72 m (worse)
                  C3 mean path per command (from Boston) L2@3s 9.23 m;
                  isolated C3 L2@3s 10.71/10.31 m (worse); pipeline C3 10.9-12.0 m
  boston_c23 (train): C2 0.85 m vs stay-still 2.72 m; C3 train L2@3s 2.1-4.7 m.
  => C2 and C3 learn Boston but do not generalise to Singapore; on the test set
     they do not beat trivial baselines. The H1 numbers above are NOT yet
     meaningful. No campaign until this is fixed.
LIKELY CAUSES: (1) overfitting: 100 training scenes, no validation/early stop,
  no augmentation; (2) Boston->Singapore shift: mean ego speed 3.7 vs 5.2 m/s;
  (3) C3 has no ego speed/history (report design, Zhai/Li), so it cannot know
  how fast to drive and is bounded by the mean-path baseline.
STATUS: paused for a user decision on fixes (see conversation).

## 2026-10-03 — Stage 1b: overfitting fixes + two C3 variants (user decision)
USER DECISION: train BOTH C3 variants on the same C2: "noego" (report design, no
  ego speed/history) and "ego" (+ ego past path at i-1..i-3, 1.5 s). Keep the one
  that beats the baselines AND uses the objects (object-use check).
FIXES (mlpipe): Boston validation split = every 7th boston_c23 scene by name
  (86 train / 14 val scenes); early stopping on Boston val (patience 8 C2, 10 C3),
  best checkpoint by Boston val (never test); mirror + rotation (+-10 deg)
  augmentation (command swapped on mirror); enable_nested_tensor=False (MPS eval
  crash). Isolated setting trained once (does not depend on C1).
  New runner mlpipe/run_stage1b.py stores the baselines next to every result:
  C2 stay-still ADE on the same scored objects, C3 mean Boston path per command.
SELF-TESTS: all passed (C3 noego memorises 0.25 m without augmentation; C3 ego
  with augmentation 1.12 m vs mean path 4.50 m; augmentation preserves distances).
SMOKE (6 train / 2 val / 3 test scenes): end-to-end OK, outputs deleted.
FULL RUN: run_stage1b --c1-seed 0 then 1, output results/stage1b/, models/stage1b/.
STATUS: RUNNING.

## 2026-10-03 — Stage 1b run STOPPED: C2 did not learn motion; C2 redesigned
The first Stage 1b run stopped itself early on the isolated C2: best Boston-val
minADE 2.440 at epoch 0, worse afterwards (2.88 at epoch 5) while train improved.
REFERENCES computed (GT identities, reference only, not a pipeline component):
  split            objects  stay-still  constant velocity from previous keyframe
  boston train     28,006   2.781 m     0.653 m
  boston val        4,736   2.306 m     0.579 m
  singapore test    8,273   4.481 m     1.008 m
=> C2 (2.44 m) was no better than "stay still": it could not find each object's
   own earlier detections in the scene-level set, so it learned no motion.
FIX: C2 now follows AutoBot-Ego's target-centred view. For every target object,
  each detection of the last 4 keyframes is described relative to the target
  (offset /5 m, distance, time offset, class, same-class flag, confidence); a
  2-layer encoder runs over this target-centred set; mode-seed decoder as before.
  Association is still learned by attention (no tracker). 736,799 parameters.
  New self-test: train on 12 synthetic scenes, test on 6 unseen ones with other
  speeds/headings: minADE 2.50 m vs stay-still 9.63 m (PASS). All self-tests pass.
Partial models/results of the stopped run deleted (isolated C2 had the old
  design); its log kept as results/stage1b_run_STOPPED_oldC2.out.
FULL RUN restarted (user: "move ahead"): run_stage1b --c1-seed 0 then 1.
STATUS: RUNNING.

## 2026-10-03 — Stage 1b moved to Kaggle (user: "skip local entirely, full run on kaggle")
Local run stopped at C2 epoch 2 of the isolated setting (val minADE already 1.090
vs stay-still 2.306 on Boston val, so the target-centred C2 learns motion).
Local partial outputs deleted; its log kept as results/stage1b_run_STOPPED_local.out.
CODE: train.device_for now prefers CUDA, then MPS, then CPU.
DATASET: ifty1011/labsd-e1-mlpipe (private) = e1_new_pkg.tar with mlpipe/*.py and
  data/stage1/{boston_c23,singapore_test}.json (18 MB).
KERNEL: ifty1011/labsd-e1-stage1b v1 (E1_new/stage1b_kaggle/), T4. Cells: extract
  package -> self-tests -> run_stage1b --c1-seed 0 -> --c1-seed 1 -> list outputs.
  Outputs: /kaggle/working/E1_new/{models,results}/stage1b/ (download to
  E1_new/models/stage1b and E1_new/results/stage1b; reuse, never retrain).
STATUS: RUNNING.

## 2026-10-03 — Stage 1b COMPLETE on Kaggle (v2, 2365 s, T4); models valid
v1 ERROR: Kaggle unpacked the uploaded .tar, kernel only looked for the .tar file.
v2: accepts the unpacked folder or the .tar. Self-tests passed on Kaggle (50 s).
Timing: seed 0 1484 s (isolated C2 638 s, others 140-220 s; C3 33-69 s each),
  seed 1 824 s (isolated reused). 15 models + manifest downloaded to
  models/stage1b (sha256 all verified); results to results/stage1b (+ kernel.log).
BOSTON VAL (early stopping): isolated C2 minADE 0.364 (constant-velocity ref 0.579,
  stay-still 2.306); pipeline C2 2.16-2.32; C3 noego 7.3-8.1; C3 ego 2.72-2.85 m.

TEST (singapore_test, 1720 anchors). Baselines: C2 stay-still 4.477 m (GT objects),
  5.39-5.85 m on the detection-matched objects; C3 mean path per command L2@3s
  9.251 m, collision 0.177.
  C2: isolated on GT 0.771 m (beats stay-still 4.48 and the constant-velocity ref
      1.008); pipeline 2.39-2.95 m vs stay-still 5.39-5.85 on the same objects. PASS.
  C3 noego (report design): L2@3s 9.3-12.5 m -> does NOT beat the mean-path
      baseline -> DROPPED (user rule: keep the variant that beats the baselines
      and uses the objects).
  C3 ego (+1.5 s ego past path): L2@3s 3.70-4.16 m, collision 0.048-0.079 (vs 9.25 /
      0.177). Object use 0.73-1.65 m (threshold 0.5 m) -> PASS, but objects move
      its plan much less than for noego (3-6 m). KEPT.
  isolation_exact = true for every model.

H1 (C3 ego), old C1 -> new C1, with random variation (swap old C1 seed):
  seed  setting       delta2 minADE  noise    delta3 L2@3s  noise
  0     isolated      -0.147         +0.059   +0.031        +0.018
  0     old-pipeline  +0.133         +0.112   -0.020        -0.013
  1     isolated      -0.170         -0.059   +0.021        -0.018
  1     old-pipeline  -0.054         -0.098   -0.001        +0.001
  new-pipeline vs old-pipeline (own C1): C3 3.705 vs 4.163 (s0), 3.697 vs 4.000 (s1).
READING: seed 0 shows the H1 pattern for C2 (the C2 trained on the old C1 gets
  worse with the new C1, +0.133, while the C2 trained on GT gets better, -0.147),
  but the old-pipeline C2 is equally hurt by a C1 seed swap (+0.112), and seed 1
  does not repeat it. Planning (delta3) moves by at most 0.03 m. Two updates are
  not enough: H1 needs a campaign of many C1 updates.

## 2026-10-04 — Pushed f4178ac; Stage 2 update campaign started on BOTH accounts
COMMIT f4178ac: mlpipe, Stage 1/1b kernels, results, run logs, model manifests.
  data/ (18 MB export) and models stay local (gitignored).
NEW mlpipe/campaign.py: Evaluator runs one C1 update through the SAVED Stage 1b
  C2/C3 (isolated and old-pipeline; C3 ego and noego), no retraining; H2 C1
  measures (moved from analysis/) + C2 output change. VERIFIED locally (eval
  only, CPU, 20 s): reproduces Stage 1b delta2/delta3 exactly (6 decimals).
INPUTS: private dataset <acct>/labsd-e1-campaign-inputs on both accounts
  (campaign_inputs.tar, 66 MB: mlpipe, singapore_test export, Stage 0 C1 models,
  Stage 1b C2/C3 models + manifests).
KERNELS: E1_new/stage2_campaign/acct{1,2}/ -> ifty1011/labsd-e1-stage2-campaign
  (base old C1 seed 0) and rai73416/labsd-e1-stage2-campaign (base seed 1), T4.
  Matrix per base (same as the earlier campaign): data {100,50,25}% x seed {1,2,3}
  at 10 epochs + epochs {5,20} x seed {1,2,3} at 100% = 15 updates; 30 in total.
  Subsets: deterministic image list per (fraction, seed) in a txt file (no copies).
  Per update: fine-tune (last.pt, val=False) -> mAP on singapore_test -> detections
  -> Evaluator -> save model + manifest + campaign_base_s{X}.json immediately.
  Random-variation reference (base vs other old C1 seed) stored first.
STATUS: RUNNING (~2.5 h expected per account).

## 2026-10-04 — Stage 2 campaign COMPLETE on both accounts (30 updates)
Results: results/stage2/campaign_base_s{0,1}.json, per-update table
results/stage2/campaign_rows.json. (Model download of acct2 hit the 30 min
background limit; results JSONs downloaded; models to be fetched separately.)

H1 (P1, C2/C3 of Stage 1b, no retraining):
  delta1 > 0 (C1 better) in 30/30 updates.
  C2 trained on the OLD C1 (old-pipeline): prediction error increases (delta2>0)
    in 23/30 updates (base seed 0: 15/15, +0.05..+0.28 m; base seed 1: 8/15).
  C2 trained on GROUND TRUTH (isolated): delta2>0 in only 5/30.
  old-pipeline delta2 > isolated delta2 in 30/30 updates.
  Random variation (swap old C1 seed): old-pipeline delta2 +0.112 (base 0) and
    -0.098 (base 1); in base 0, 11/15 updates exceed +0.112.
  => For the prediction module, entangled enhancement occurs in most updates,
     and it is tied to C2 having been trained on the old C1 (learned-mistakes
     pattern, H1a): the GT-trained C2 almost always improves.
  C3 (ego variant): old-pipeline delta3>0 in 18/30, mean +0.008 m (tiny);
    isolated delta3>0 in 29/30, mean +0.053 m (1-3% of L2@3s ~4 m). Planning
    error slightly increases even for the GT-trained C3, so this small planning
    effect is not tied to learned mistakes (closer to H1b / input-change).
H2 (label-free C1/C2 output change as symptom of harm to C2), 30 updates:
  pooled Spearman with delta2: C2 same-object shift +0.68 (AUROC 0.81), type mix
  JS -0.51, others |rho| <= 0.29; delta1 +0.04 (AUROC 0.66).
  BUT within each base the C2-shift correlation is +0.29 and -0.28: the pooled
  value is driven by the difference between the two base models (base 0 has
  both larger shifts and larger harm). No measure predicts harm consistently
  within a base. => H2 not supported yet with 15 updates per base.

## 2026-10-04 — Stage 2 results archived; model download skipped (user decision)
User: do not download the 30 campaign C1 models (they stay in the Kaggle kernel
outputs ifty1011/ and rai73416/labsd-e1-stage2-campaign); download results only.
Saved: campaign_base_s{0,1}.json, campaign_rows.json, manifest_base_s{0,1}.json
(per-update settings, test mAP, sha256), and the two kernel logs (downloaded by
the user from the Kaggle UI) as log_base_s{0,1}.log.gz.
Kernel run times: base seed 0 (acct1) 9096 s; base seed 1 (acct2) 8002 s.

## 2026-10-04 — Strict count of entangled enhancement (above random variation)
Threshold = largest |change| from swapping only the old C1 seed (both bases, both
settings): prediction 0.112 m, planning (C3 ego) 0.018 m. Old-pipeline C2/C3:
  prediction harm (delta2 > 0.112): 11/30 (base seed 0: 11/15, base seed 1: 0/15)
  planning harm  (delta3 > 0.018): 13/30
  either: 21/30 updates
Same threshold for the C2/C3 trained on GROUND TRUTH:
  prediction harm 0/30, planning harm 29/30.
READING: the prediction harm is tied to C2 having learned the old C1 (11 vs 0),
  i.e. learned mistakes, but only from base seed 0. The planning increase also
  appears for the GT-trained C3 (29/30), so it is not learned mistakes; it is a
  small effect (0.02-0.13 m of ~4 m).
CAVEAT: the noise threshold comes from ONE seed swap per base. A paired
  bootstrap over the 50 test scenes would give a proper interval per update;
  it needs the per-update detections (in the Kaggle kernel outputs).

## 2026-10-04 — Meeting 7 report pushed (965dc7d); Stage 3 bootstrap started
Report: Reports/Meeting 7/Report_Ifty_Meeting7.{tex,pdf} (8 pages, academic format,
checked against SENSEI_WRITING_MASTER.md). Email to sensei postponed (user).
STAGE 3 (user: "move to the next part"): paired scene bootstrap per update to
replace the single seed-swap threshold. NEW mlpipe/bootstrap.py: per-scene sums
of object minADE and anchor L2@3s (C3 ego); 2000 resamples of the 50 test scenes;
95 % interval of delta2 and delta3; harm = whole interval above zero.
Local check (eval only): point values equal the campaign deltas exactly; interval
for the Stage 1b update: old-pipeline delta2 +0.133 [-0.115, +0.427].
CODE: private datasets <acct>/labsd-e1-mlpipe-code (mlpipe with bootstrap.py).
KERNELS (CPU, no internet): ifty1011 and rai73416 /labsd-e1-stage3-bootstrap
(E1_new/stage3_bootstrap/acct{1,2}); inputs campaign-inputs + mlpipe-code +
kernel_sources <acct>/labsd-e1-stage2-campaign (the 15 detection files).
Output bootstrap_base_s{0,1}.json. STATUS: RUNNING.

## 2026-10-04 — Stage 3 bootstrap COMPLETE (both accounts, CPU)
Results: results/stage3/bootstrap_base_s{0,1}.json (+ kernel logs).
Paired scene bootstrap (2000 resamples of the 50 test scenes, 95 % interval);
significant = whole interval above zero (harm) or below zero (benefit).
Interval width: about +-0.2 to +-0.3 m for delta2, +-0.05 to +-0.1 m for delta3.
  C2/C3 trained on the OLD C1:     prediction harm 3/30 (all base seed 0:
                                   f100_e5_s2 +0.284, f100_e5_s3 +0.266,
                                   f25_e10_s2 +0.284); planning harm 1/30.
  C2/C3 trained on GROUND TRUTH:   prediction harm 0/30 (benefit 2/30);
                                   planning harm 10/30.
  Seed swap (random variation) is not significant in any setting.
READING: per update, the 50 test scenes give wide intervals and only 3 updates
  show significant prediction harm. The consistent pattern across updates is
  stronger than any single update: old-pipeline delta2 > GT-trained delta2 in
  30/30 updates, and delta2 > 0 in 15/15 updates from base seed 0. The updates
  share one base model and one test set, so they are not independent; a
  pooled test should account for that. The planning increase is more often
  significant for the GT-trained C3 (10/30) than for the old-pipeline C3 (1/30).
