# Scaling Up the Experiment: Before vs. Now

**A plain-language summary of what changed when we moved from a tiny test to a large one.**

*(Based on Kernel A and Kernel B. The class-balanced runs, Kernel C, are still
computing and will be added when ready.)*

---

## 1. The one big change

Before, we tested our idea on a **very small** slice of data (just 3 scenes (about 121 frames)).
Now we have tested it on a **large** slice (2,020 frames from 50 scenes).

This is the whole story. Everything below is a consequence of using proper data
instead of a tiny sample.

---

## 2. How much data — before vs. now

A "scene" is a ~20-second driving clip. Each scene contains many camera frames
(images). What matters for training and testing is the number of **images**.

| | Before (mini) | Now (full trainval) |
|---|---|---|
| Boston training scenes | ~6 scenes | **280 scenes = 11,087 images** |
| Singapore fine-tune data | 119 images | **229 scenes** |
| **Evaluation (test) set** | **3 scenes (~121 frames)** | **50 scenes = 2,020 images** |

So our evaluation went from **3 scenes (~121 frames)** to **50 scenes (2,020 frames)** —
about **17 times more** data to test on. This is the difference between an
anecdote and a real measurement.

---

## 3. What the pipeline is (unchanged)

The system is the same three-part self-driving pipeline:

- **C1 — Perception:** a camera detector (YOLOv11) that finds cars, pedestrians,
  etc.
- **C2 — Prediction:** predicts where those agents will move.
- **C3 — Planning:** decides the car's driving path.

We **retrain only C1** (on new-city data) and watch what happens downstream to
C2 and C3. That setup did not change.

---

## 4. The key result — before vs. now

### The problem we had before

Our core idea is **"entangled enhancement"**: you make the detector *better on
its own score*, but the whole system gets *worse*. To show this, the detector's
own score must go **up**.

**Before (3 scenes):** when we retrained the detector, its own score went
**DOWN** in all 15 tries. This was a side effect of having too little training
data (only 119 images) — the detector over-fitted. Because the score went down,
we **could not** show entangled enhancement directly. We had to use a special
trick (class balancing) just to force the score up.

### What happens now

**Now (2,020 frames):** with proper training data, the detector's own score goes
**UP** in all 15 tries — the small-data problem is gone. And in **6 of those 15
tries, the planner still gets worse at the same time.**

That is exactly entangled enhancement — **the detector improves, but the driving
output degrades** — and now we show it **directly, with no trick needed.**

### Side-by-side

| What we measured | Before (3 sc/121 fr) | Now (50 sc/2020 fr) |
|---|---|---|
| Detector improves after retraining? | 0 of 15 | **15 of 15** |
| Prediction improves? | mixed | **15 of 15** |
| **Entangled enhancement seen directly?** | **0 of 15 (needed a trick)** | **6 of 15 (naturally)** |
| Collisions ever happen? | Never (0) | **Yes (2%)** |
| Plan shift range | 1.33 – 7.80 m | 0.78 – 1.50 m |

---

## 5. The baseline numbers now (the "before-retraining" system)

Measured on 2,020 frames:

| Metric | Value | Meaning |
|---|---|---|
| C1 detector mAP | 0.137 | detector works |
| C2 prediction error (isolated) | 1.12 m | clean prediction |
| C2 prediction error (in pipeline) | 7.93 m | error inherited from detector — the cascade |
| C3 planning error (isolated) | 4.64 m | planner alone |
| C3 planning error (in pipeline) | 5.55 m | **worse than isolated → the cascade is real** |
| Collision rate | 2% | collisions now occur |

The important line: **C3 in the pipeline (5.55 m) is worse than C3 alone
(4.64 m).** That gap is the cascade — the planner is hurt by errors flowing down
from perception. This holds cleanly across 50 scenes now, not just 3.

---

## 6. What this means for the paper

Three honest takeaways:

1. **The main claim is now much stronger.** "Entangled enhancement happens in a
   real self-driving pipeline" is shown by **6 ordinary retrainings on proper
   data**, not by a trick on 3 scenes.

2. **One old finding disappears.** The "detector score drops after retraining"
   result was a **small-data artifact.** With real data it vanishes. That part of
   the old paper must be removed or reframed.

3. **A new capability appears.** Collisions now occur (they never did at n=3), so
   the safety part of our proposed method (CARA) becomes testable.

In short: the big data jump made the paper **more convincing and more honest.**
What used to be a weakness (tiny sample, a workaround) is now a clean result at
real scale.

---

## 7. What is still coming

The **class-balanced runs (Kernel C)** are still computing on two accounts in
parallel. At the small scale, these carried the headline. At full scale, Kernel B
already carries it — so Kernel C now just adds a supporting check: *"does class
balancing add even more entangled-enhancement cases?"* We will fold those numbers
in when they finish.
