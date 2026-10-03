# E1 (new stage): ML pipeline, 350-scene subset

Tests H1 (existence) and H2 (symptom) of the Meeting 6 report on a pipeline in
which all three modules are ML components. Earlier experiments are in
`../E1_kaggle_old/`.

## Layout

```
E1_new/
├── EXPERIMENT_LOG.md     # every action, mandatory
├── subset_350/           # Kaggle kernel: builds the 350-scene subset (account 1)
├── subset_350_acct2/     # same kernel for account 2 (rai73416)
├── stage0_c1/            # Stage 0: old C1 x 2 seeds + one update each
├── analysis/             # local analysis scripts (H2 measures, ...)
├── models/               # downloaded trained models + manifests (gitignored)
└── results/              # downloaded kernel outputs, logs and summaries
```

## Rules

- Every kernel saves each trained model with a `manifest.json`. Models are
  downloaded to `models/<stage>/` and reused, never retrained.
- Kernels that load the subset must use the placeholder-map fix (see Stage 0).
- The labsd code package used on Kaggle is `ifty1011/labsd-src` (source in
  `../E1_kaggle_old/src/`).
