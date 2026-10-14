# Revision plan for CS_seminar_IFTY (v0 → v1): remaining work

Temporary working file. Only the changes that are still open are listed. Part A is the plan. Part B is the paper as it is now, with every remaining change marked on the text.

Already done (not listed below): title (C1); all of sensei's strike-outs and his inserted words (C2, C4–C6, C8–C10, C12–C14, C18, C20–C23, C30, C36, C39, C41, C42, C51, C52); contribution list deleted (C17, the new text is still open); every "harm" word replaced (C19, C47, C56, C63, C66, C68, C70); "we" sentences (E2); C11, C67, C75, C76.

How to read the marks in Part B:
- ~~struck text~~ = delete
- **bold text** = insert
- `[C12]` = sensei's comment number in SENSEI_COMMENTS_v0.md
- `[G3]` = a common word change from Part A, Step 1
- `> NOTE` = what sensei asks for where he gave no exact wording (to be written)

---

# Part A: The plan

## Decisions needed first

1. **"Update".** Keep "update" for one retraining of $C_1$. Define it once: "An update retrains $C_1$ on new data and gives the updated model $C_1'$."
2. **Number of hypotheses.** Keep the paper's two (H1 with two sentences, H2 for detection), or use three like the presentation (H1 occurrence, H2 cause, H3 detection). Part B keeps two; the numbering changes if we choose three.

## Step 1: Common word changes (apply everywhere)

Done: G2 ($C_1$/$C_1'$), G3 (GT training / $C_1$-output training / retraining on the output of $C_1'$), G5 (seed variation).

| ID | Find | Replace with | Why |
|---|---|---|---|
| ~~G2~~ | ~~old $C_1$ / new $C_1$~~ | done: $C_1$ / $C_1'$ everywhere; "the $C_1$ with seed 0/1" for the two seeds; "the perception model before the update" in the abstract | C26, C40, C53 |
| G6 | ML component(s) | ML model(s) | C6, C13 |
| G6 | the perception/prediction/planning module $C_i$ | the perception/prediction/planning model $C_i$ | C39, C41, C42 |
| G7 | mistakes; adaptation to the mistakes | errors; the adaptation of $C_2$ and $C_3$ to the errors of $C_1$ (defined once) | C73 |
| G9 | "X, and Y" where X and Y are different facts | two sentences | C74 |
| G11 | collision rate | delete everywhere | C45 |

## Step 2: Move and restructure sections

| Now | Change to | Why |
|---|---|---|
| ~~Introduction (C15, C16, C17)~~ | done: earlier experiment deleted; gap, objective, what we did, and summary of results written | C15–C17 |
| ~~III-A: define "ego vehicle"~~ | dropped (user decision 2026-10-14): sensei struck the definition, so no definition is added anywhere | C14 |
| III-A Pipeline and Notation | first say the report focuses on the AV pipeline, then define $C_1$, $C_2$, $C_3$, $C_1'$ | C24, C26 |
| III-B Hypotheses | add the goal and research questions before the hypotheses; full labels; a reason for each | C27–C33 |
| III-C Workflow, III-D Update Procedure | move to Section IV, after the data set | C34, C38 |
| IV-D Training and Measurements | define GT and $C_1$-output training; define seed variation in plain words; delete the collision rate | C35, C45, C46 |
| V (all subsections) | one subsection per research question; open with what it tests; end with a takeaway | C48, C57, C61, C71 |
| V-D New-Pipeline Training | rename "Retraining $C_2$ and $C_3$ on the Output of $C_1'$" | C62 |
| V-E Symptom | explain why ground truth is missing, what the nine measures are, why rank correlation; rewrite or delete the last paragraph | C64, C65, C69, C71 |
| VI Conclusion | simple sentences, no "and" joins, explain "adaptation" | C72–C74 |

New outline:
1. Abstract (rewritten)
2. I. Introduction: MLS and entangled enhancement → AV pipeline → existing studies and their limitation → objective and what we did → summary of results
3. II. Related Work
4. III. Methodology: A. Target pipeline and notation, B. Objective, research questions and hypotheses
5. IV. Experimental Setup: A. Data, B. ML models ($C_1$, $C_2$, $C_3$), C. Training modes of $C_2$ and $C_3$, D. Workflow and update procedure (Fig. 1, Algorithm 1), E. Measurements (seed variation, bootstrap), F. Check of $C_2$ and $C_3$
6. V. Results and Discussion: A. RQ1 and RQ2 for $C_2$, B. RQ1 and RQ2 for $C_3$, C. Retraining $C_2$ and $C_3$ on the output of $C_1'$, D. RQ3: detection without ground truth
7. VI. Conclusion

## Step 3: Parts that need new writing

1. Abstract: gap, objective, evaluation procedure, results (C3, C7).
2. Introduction: gap, difference from existing studies, objective, what we did, summary of results (C15–C17).
3. Goal and research questions before the hypotheses, reasons, new labels, new H2 wording (C27–C33).
4. Plain sentences for the $C_3$ inputs and the object-use check (C43, C44).
5. Seed variation in plain words (C46).
6. Why we compare with baselines (C49).
7. Purpose sentence and takeaway for every result subsection (C48, C50, C54, C55, C57–C61, C64, C65, C69, C71).
8. Conclusion sentences (C72–C74).

## Step 4: Missing information

- CPU and RAM of the Kaggle session (C37). Not recorded in the logs.

## Page-count cuts applied 2026-10-14 (all supported by sensei's marks)
- Algorithm 1 removed; IV-C "Update Procedure" now states the procedure in words (sensei's only marks on the algorithm were the training names).
- V-B seed-swap paragraph deleted (C55 "Not sure where are you talking about"); the two kept sentences now point to Fig. 2 (C54).
- Collision-rate sentences deleted in IV-F and V-A (C45); V-A gives the planning error against the baseline instead.
- V-E last paragraph deleted (C68, C69: "harm"/"random variation"/"better symptom" made the discussion unclear).
- mistakes → errors (G7), "ML components" → "ML models" in Table I caption (G6).

---

# Part B: The paper now, with the remaining changes marked

## Abstract

Done (C2–C7, G2, G5 applied). Rewritten in sensei's order: definition, limitation of existing research, this report and its objective, evaluation procedure, results as answers.

## I. Introduction

Done (C8–C17 applied). Earlier experiment and contribution list deleted; P2 now ends with the existing studies and their limitation; P3 gives the objective and what we did; P4 gives the summary of the results. Open here: G6 (component → model) in P1, and the "ego vehicle" definition moves to III-A.

## II. Background and Related Work

Done and frozen (user 2026-10-14: keep as it is). (C18–C23 applied; "Entangled enhancement can have two causes" removed because sensei struck the sentence in that place; component → model, mistakes → errors, module → model throughout the section).

## III. Methodology

Done (C24–C35 applied). III-A "Target Pipeline and Notation": focus sentence, then $C_1$, $C_2$, $C_3$ defined one by one; "new city" removed. III-B "Research Questions and Hypotheses": goal, RQ1–RQ3 with reasons, H1 (occurrence), H2 (cause), H3 (detection), descriptive labels. III-C Workflow and III-D Update Procedure moved to Section IV after Data (C34). Training modes renamed everywhere to GT training / $C_1$-output training (C35); "random variation" renamed and explained as seed variation everywhere (C46); V-D renamed "Retraining $C_2$ and $C_3$ on the Output of $C_1'$" (C62). Hypothesis numbering: the detection hypothesis is now H3 in Fig. 1, Algorithm 1, V-E and the Conclusion.

## IV. Experimental Setup

**Opening**

This section presents the data set and the ML models used for experiments. We trained all models on one Tesla T4 graphics card **(add CPU and RAM)**. `[C37]`

**A. Data** (first part of Section IV `[C38]`)

We use 350 of the 850 scenes of nuScenes [3], which are driving clips of about 20 s from Boston and Singapore, with the full-size images of the front camera only (4.6 GB). **We split the scenes into four parts.** Boston 1 (100 scenes) trains $C_1$~~, and~~**.** Boston 2 (100 scenes) trains $C_2$ and $C_3$, with 14 of the scenes kept for validation. Separate scenes give $C_2$ and $C_3$ the realistic ~~mistakes~~ **errors** of $C_1$ [12]. Singapore 1 (100 scenes) is used for the updates**, as data from a new city after deployment**~~, and~~**.** Singapore 2 (50 scenes, 2,020 images) is used for testing. A keyframe is an image with labelled objects, recorded twice per second. The test set has 1,720 test samples, one for every keyframe with 3 s of future. `[C25, G7, G9]`

**B. Updates of the Perception ~~Module~~ Model**

We use YOLOv11n [13] as a perception model ($C_1$), which is trained for 10 passes over Boston 1 with the random seeds 0 and 1. The two ~~old~~ **original** $C_1$ reach an mAP50 of 0.109 and 0.117 on Singapore 2. In each update, we retrain ~~an old~~ **an original** $C_1$ on Singapore 1 with one of 15 settings. Nine settings use 100 %, 50 % or 25 % of the images with 10 passes~~, and~~**.** Six use all images with 5 or 20 passes, each with the seeds 1, 2 and 3. The 15 settings applied to both ~~old~~ **original** $C_1$ give 30 updates. The test scenes are never used to choose a model. `[C40, G2, G9]`

**C. ML Prediction and Planning ~~Modules~~ Models**

AutoBot-Ego [6] is employed as a prediction model ($C_2$). For each current object, a transformer reads the objects of the last four keyframes relative to that object and outputs six possible paths of 3 s, without a hand-written rule that links objects across images [13]. For the planning model ($C_3$), we employ PlanT [10]. ~~Every object is one input with the current position, the most likely path from $C_2$, the class and the confidence.~~ `[C43]` Two further inputs hold the driving command and the path of the ego vehicle over the last 1.5 s, which we call the ego path. The command is left, straight or right [11]~~, and~~**.** A recurrent decoder outputs six points of the planned path. `[G9]`

> NOTE `[C43]`: "not interpretable". For example: "$C_3$ receives one input for each object. This input contains the position of the object, its most likely path predicted by $C_2$, its class and its confidence."

We train both models on Boston 2 with early stopping on the validation scenes and with mirror and rotation augmentation. ~~An input of the ego path can let a planning module ignore the objects [14], and we check whether $C_3$ uses the objects. Removing the objects from the input of $C_3$ moves the planned position at 3 s by 0.73 to 1.65 m on average, which is above the threshold of 0.5 m fixed before the experiment.~~ `[C44]`

> NOTE `[C44]`: "Moderate the sentences. Hard to follow." For example: "A planning model with an ego-path input can ignore the objects [14]. We therefore check whether $C_3$ uses the objects. We remove the objects from the input of $C_3$. The planned position at 3 s then moves by 0.73 to 1.65 m on average. This shift is above our threshold of 0.5 m. $C_3$ therefore uses the objects."

**D. Training and Measurements**

We train $C_2$ and $C_3$ once with ~~isolated~~ **GT** training and once with ~~old-pipeline~~ **$C_1$-output** training for each ~~old~~ **original** $C_1$~~, and~~**.** The trained models stay fixed during all updates. The prediction error is computed over the objects found by $C_1$ that are matched one to one with a ground truth object of the same class. ~~The collision rate is the fraction of test samples in which the planned path overlaps an object.~~ `[C45]` The outputs of $C_2$ and $C_3$ on the ground truth stayed exactly the same before and after every update, which confirms that only the output of $C_1$ changed. The ~~random variation~~ **seed variation** `[C46]` is $\tau_2 = 0.112$ m and $\tau_3 = 0.018$ m. For the bootstrap, we draw the 50 test scenes 2,000 times with replacement. `[G2, G3, G5, G9]`

## V. Results and Discussion

> NOTE `[C48]`: "not structured well … should be along with your research questions." Each subsection opens with the research question it answers and ends with a takeaway.

**Opening**

In this section, we first check that $C_2$ and $C_3$ work, because a count of updates that increase the error is meaningful only for working ~~modules~~ **models**. We then report and discuss the results. Every update made $C_1$ more accurate, with $\delta_1$ between 0.048 and 0.118. `[G6]`

**A. Validity of the ML ~~Components~~ Models**

> NOTE `[C49]`: explain why: if $C_2$ or $C_3$ were no better than a simple rule, a change of their error would say nothing about the update.

We compare $C_2$ and $C_3$ with simple baselines on the test set (Table I). $C_2$ beats the baseline in which every object stays still. The variant of $C_3$ without the ego path does not beat the baseline that drives the average Boston path for each command. We use the variant with the ego path, which has ~~a collision rate of 0.05 to 0.08 against 0.18~~ **an error of 3.70 to 4.16 m against 9.25 m** for the baseline. `[C45]`

Table I caption: "ML ~~components~~ **models** against simple baselines on the test set (m)". `[G6]`

**B. Prediction ~~Module~~ Model** (RQ1 and RQ2 for $C_2$)

> NOTE `[C48]`: open with the question: "We examine whether an update increases the prediction error, and whether the training mode matters."

Fig. 2 shows $\Delta_2$**, the increase of the prediction error,** `[C50]` for every update in two training modes. In all 30 updates, $\Delta_2$ after ~~old-pipeline~~ **$C_1$-output** training is larger than $\Delta_2$ after ~~isolated~~ **GT** training. The difference is largest for the updates from the ~~old~~ **original** $C_1$ with seed 0, where $\Delta_2$ after ~~old-pipeline~~ **$C_1$-output** training lies between +0.049 and +0.284 m with a mean of +0.165 m. `[G2, G3]`

Table II counts the updates with an increase at the three levels of evidence. We observe that, above ~~random~~ **seed** variation, the prediction error increased in 11 updates after ~~old-pipeline~~ **$C_1$-output** training and in none after ~~isolated~~ **GT** training. All 11 updates start from the ~~old~~ **original** $C_1$ with seed 0. With 50 test scenes, the bootstrap interval of $\Delta_2$ spans about ±0.25 m. Only 3 updates after ~~old-pipeline~~ **$C_1$-output** training and none after ~~isolated~~ **GT** training have an interval above zero. `[G2, G3, G5]`

The comparison results characterize the first cause of entangled enhancement. Both versions of $C_2$ receive the same output of ~~the same new $C_1$~~ **$C_1'$** `[C53]`~~, and~~**.** Only the version that learned from ~~the old~~ **the output of** $C_1$ becomes worse. `[C54]` ~~The $C_2$ trained on the old $C_1$ with seed 0 also becomes worse by 0.112 m when the old $C_1$ is replaced by the other old $C_1$, although the two old $C_1$ differ by only 0.009 in mAP50. This $C_2$ reacts to the different mistakes of the other old $C_1$, because the accuracy of the two old $C_1$ is almost the same. The $C_2$ trained on the old $C_1$ with seed 1 improves by 0.098 m under the same replacement. No update from seed 1 increases the error of this $C_2$ by more than random variation. The increase after an update is larger for the old $C_1$ with the stronger adaptation.~~ `[C55]`

> NOTE `[C54]`: "Where should I look at?" Point to the figure: "(Fig. 2, all red points lie above the blue points)".
> NOTE `[C55]`: "Not sure where are you talking about." The seed-swap result is not in any figure or table. Explain it with a pointer, move it to the discussion in plain words, or delete it.
> NOTE `[C57]`: end with a takeaway, for example: "As a result, we confirm H1 for $C_2$. The increase comes from the adaptation of $C_2$ to the errors of $C_1$."

Fig. 2: caption "Red circles show ~~old-pipeline~~ **$C_1$-output** training, blue squares show ~~isolated~~ **GT** training, and the grey band shows the ~~random~~ **seed** variation." Labels "old $C_1$ seed 0/1" → "original $C_1$ with seed 0/1".
Table II: "op" → "$C_1$-output", "iso" → "GT", "Above random variation" → "Above seed variation".

**C. Planning ~~Module~~ Model** (RQ1 and RQ2 for $C_3$)

> NOTE `[C58]`: "What would you like to confirm?" Open with: "We examine whether the increase reaches the end of the pipeline, the planned path of $C_3$."

The planning error changed little. The mean $\Delta_3$ lies between −0.006 and +0.058 m~~, and~~**.** The largest value is 0.13 m against a planning error of about 4 m. The increase appears more often after ~~isolated~~ **GT** training than after ~~old-pipeline~~ **$C_1$-output** training at every level in Table II. Adaptation to the ~~mistakes~~ **errors** of ~~the old~~ $C_1$ cannot explain an increase after ~~isolated~~ **GT** training. `[G2, G3, G7, G9]`

$\Delta_3$ is small because $C_3$ plans mainly from the ego path. `[C59]` The objects move the planned position at 3 s by only 0.73 to 1.65 m. ~~The more frequent increase after isolated training points to the second cause.~~ `[C60]` The $C_3$ trained on the ground truth receives the predicted paths of $C_2$ only at test time. A change in the objects that ~~the new $C_1$~~ **$C_1'$** finds can therefore affect this $C_3$ more than the $C_3$ trained on the output of ~~the old~~ $C_1$. `[G2]`

> NOTE `[C59]`: "could not understand why?" Explain: removing all objects moves the plan by only 0.73 to 1.65 m, while the planning error is about 4 m. Most of the plan therefore comes from the ego path.
> NOTE `[C60]`: "?" Delete the claim about the second cause, or explain it step by step.
> NOTE `[C61]`: end with a conclusion against the motivation: "The planning error increases by at most 0.13 m. The first cause cannot explain this increase."

**D. ~~New-Pipeline Training~~ Retraining $C_2$ and $C_3$ on the Output of $C_1'$** `[C62]`

We test whether retraining $C_2$ and $C_3$ ~~together with $C_1$~~ **on the output of $C_1'$** lowers the prediction error and the planning error below their values before the update. ~~New-pipeline training retrains $C_2$ and $C_3$ on the output of the new $C_1$.~~ We apply ~~it~~ **this retraining** to two further updates, one from each ~~old~~ **original** $C_1$ with all images and 10 passes. After ~~old-pipeline~~ **$C_1$-output** training, the prediction error after the update is 2.75 and 2.61 m~~, and~~**.** The planning error is 4.14 and 4.00 m. After ~~new-pipeline training~~ **the retraining**, the prediction error is 2.39 and 2.45 m~~, and~~**.** The planning error is 3.71 and 3.70 m. `[C62, G2, G3, G9]`

~~New-pipeline training~~ **The retraining** lowers the prediction error by 0.36 and 0.16 m ~~and~~**. It lowers** the planning error by 0.44 and 0.30 m. The errors also fall below the errors before the update (2.62 and 2.66 m, 4.16 and 4.00 m)~~, and the update becomes an improvement of the whole pipeline~~. This result supports the retraining of downstream ~~components~~ **models** that Wang and Machida model [8, 28]. This retraining has a cost. A symptom of entangled enhancement is therefore useful. `[G3, G6, G9]`

**E. ~~Symptom of Entangled Enhancement~~ Detection without Ground Truth** (RQ3)

> NOTE `[C64, C65]`: open with why: "After deployment, new data have no ground truth, and the error cannot be measured." Then list the nine measures in plain words (number of objects per image, mix of classes, confidence values, positions, objects found by only one version, length of predicted movements, shift of predicted positions). Then why rank correlation: "A useful measure is larger for updates with a larger $\Delta_2$."

Nine measures of output change need no ground truth, seven on the output of $C_1$ and two on the output of $C_2$. `[C64]` The rank correlation `[C65]` with $\Delta_2$ is close to 1 when a larger measure always comes with a larger $\Delta_2$. The AUROC is the probability that an update with $\Delta_2 > 0$ has a larger measure than an update with $\Delta_2 \le 0$. Over all 30 updates, the shift of the predicted position of the same object has the highest rank correlation (+0.68, AUROC 0.81). For each ~~old~~ **original** $C_1$ separately, the rank correlation is only +0.29 and −0.28. The high value over all updates comes from the difference between the two ~~old~~ **original** $C_1$. The gain $\delta_1$ has a rank correlation of +0.04. No measure consistently separates the updates that increase the prediction error from the other updates. H2 is therefore not supported. `[G2]`

~~The rank correlation of $\delta_1$ shows that the accuracy gain of $C_1$ does not indicate the increase of the prediction error. The measures on the output of $C_1$ describe the update alone. The increase depends on how strongly $C_2$ adapted to the old $C_1$. The random variation in Section V-B suggests a better symptom. The old $C_1$ whose $C_2$ was disturbed by the replacement of the old $C_1$ is also the old $C_1$ whose updates increased the error of $C_2$. A measure of how strongly $C_2$ reacts to any change of the output of $C_1$ can describe the adaptation of $C_2$ before an update is made.~~ `[C69]`

> NOTE `[C69]`: "Neither 'random variation' or 'better symptom' is understandable." Rewrite in plain words or delete. If kept: "The accuracy gain $\delta_1$ does not indicate the increase of the prediction error. The increase depends on how strongly $C_2$ adapted to the errors of $C_1$."
> NOTE `[C71]`: "Again, so what?" End with a conclusion: "As a result, none of the nine measures identifies single updates with entangled enhancement."

## VI. Conclusion

In this report, we updated the perception ~~module~~ **model $C_1$** 30 times. ~~We observe that the prediction module became worse more often and by more after old-pipeline training than after isolated training.~~ `[C72]` ~~This result supports H1 for the prediction module and points to adaptation to the mistakes of the old perception module as the cause.~~ `[C73, C74]` The planning error changed little. No measure of output change that needs no ground truth identified the updates that increased the prediction error. H2 is therefore not supported. As future work, we develop a method that detects entangled enhancement after an update of the perception ~~module~~ **model**, before the update is used and before ground truth for the new data exists. `[G6]`

> NOTE `[C72]`: "Make it more simple and straight forward." For example: "We observe that the prediction error of $C_2$ increases above seed variation in 11 updates after $C_1$-output training and in none after GT training."
> NOTE `[C73, C74]`: one message per sentence, and explain "adaptation". For example: "As a result, we confirm H1 for $C_2$. The increase comes from the adaptation of $C_2$ to the errors of $C_1$."

---

Coverage: this file lists every open comment from SENSEI_COMMENTS_v0.md plus the common word changes G2, G3, G5, G6, G7, G9 and G11.
