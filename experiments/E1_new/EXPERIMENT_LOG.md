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
