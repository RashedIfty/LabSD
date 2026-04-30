# E1 — Cascade Degradation Experiment (Kaggle)

Scaled-down version of Experiment E1 from the Meeting 3 report.

## Goal

Empirically test whether retraining C1 alone degrades C3 (pipeline) while
C3 (isolated) remains unchanged — i.e. cascade entanglement across two hops.

## Pipeline

C1 (CenterPoint) → C2 (MTP) → C3 (IDM planner, PDM-Closed substitute)

## Dataset

nuScenes v1.0-mini (~4 GB, 10 scenes). Split by location:
- Boston: training of all three components
- Singapore: domain-shift evaluation + C1 retraining

## Compute

Kaggle free tier: 2× T4 GPU, 30 GPU-hr/week budget.

## Layout

```
experiments/E1_kaggle/
├── README.md
├── kernel-metadata.json     # Kaggle notebook config
├── notebook.ipynb           # thin orchestrator (TODO)
├── configs/
│   ├── centerpoint_boston.py
│   └── centerpoint_singapore.py
├── src/                     # local development copy
│   ├── splits.py
│   ├── train_c1.py
│   ├── train_c2.py
│   ├── c3_idm.py
│   ├── eval.py
│   └── table.py
├── labsd-src/               # Kaggle dataset staging dir
│   ├── setup.py
│   ├── dataset-metadata.json
│   └── labsd/
│       └── (mirrors src/)
└── results/                 # downloaded from Kaggle
```

## Workflow

1. Edit code in `src/`.
2. Sync to `labsd-src/labsd/` and push as Kaggle Dataset.
3. `kaggle kernels push` from this directory.
4. Run All in browser.
5. `kaggle kernels output rashedifty/labsd-e1-cascade -p ./results/`.

See top-level conversation notes for full step list.
