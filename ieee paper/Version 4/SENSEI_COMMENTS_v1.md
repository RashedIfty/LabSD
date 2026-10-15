# Sensei's comments on CS_seminar_IFTY (v1, second review)

Sources:
- **Email** (Fumio Machida, 2026-10-15), with the commented report attached.
- **`CS_seminar_IFTY-v1-comment.pdf`**: 75 marks on five pages (the version sent on 2026-10-14, commit b48ab9d). 26 notes or highlights with text, 22 strike-outs, 18 insertions, the rest bare highlights.

How each comment is shown:
- **Paper:** the exact text sensei marked.
- **Sensei:** his strike, insertion or note, quoted as written.
- **After:** for a wording edit, the sentence with his edit applied. For an instruction or question, what has to change.
- `#n`: the mark number in reading order (page, left column, right column, top to bottom).

---

## Email

> Please find the attached comments. Still not far from exhaustive, but I tried to provide as much feedback as possible. Prepare an updated version before the meeting, then I will check the new version tomorrow. (Incomplete version is ok. We need further iteration to improve the quality)
>
> **I think the key is a set of RQs. You had better restructure the experiment design and result presentation corresponding to RQs** (See more details in the file). **Hypotheses are actually RQs, so you can remove them.**
>
> As the presentation can significantly change depending on the update of the report. **You should prioritize polishing the report.**

Main points of the email:
1. **Deadline:** an updated version before the meeting; incomplete is acceptable.
2. **Research questions:** the RQs are the key. Restructure the experiment design and the result presentation by RQ.
3. **Hypotheses:** remove them; the RQs replace them.
4. **Priority:** the report first; the presentation follows the report.

---

## Title and author block (page 1)

**D1. Author block (#1, instruction)**
- Paper: the author block is commented out (no author shown).
- Sensei: "Write your name, student ID, affiliation, email address, etc."

## Abstract (page 1)

**D2. Sentence 1 (#2, edit)**
- Paper: "Entangled enhancement is a defective retraining of a machine learning (ML) model that improves the updated model **performance** but fails to improve the whole system."
- Sensei: strikes "performance".
- After: "Entangled enhancement is a defective retraining of a machine learning (ML) model that improves the updated model but fails to improve the whole system."

**D3. Gap sentence (#3, #4, edit)**
- Paper: "No study has measured how often this effect occurs in a driving pipeline."
- Sensei: strikes the sentence, inserts "Although the problem is known in the literature, little has been studied in real ML pipelines."
- After: "Although the problem is known in the literature, little has been studied in real ML pipelines."

**D4. "This report …" (#5–#8, edit)**
- Paper: "This **report** empirically investigates entangled enhancement in an autonomous driving pipeline consisting of perception, prediction and planning modules."
- Sensei: "report" → "study"; inserts "real" before "entangled"; inserts "s" after "enhancement".
- After: "This **study** empirically investigates **real** entangled **enhancements** in an autonomous driving pipeline consisting of perception, prediction and planning modules."

**D5. Rest of the abstract (#9, instruction)**
- Paper: "We examine whether an update that improves the perception model increases the error of the prediction and planning models, whether their adaptation to the errors of the perception model causes the increase, and whether entangled enhancement can be detected without ground truth. …"
- Sensei: "We expect the details of the empirical investigations. To meet the expectations, first, you should explain that you trained an ML pipeline by nuScenes dataset. Next, explain you investigated the change in prediction module when the perception model is updated with a new dataset. Then explain 30 different test cases. Finally, you can explain that entangle enhancements are observed in 23 test cases over 30. After presenting the main outcome, then it is ready to explain the next part. In order to analyze the cause of entangle enhancement, you did a comparative studies, right? Abstract should organize like this way."
- After: the abstract follows his order:
  1. we trained an ML pipeline on nuScenes;
  2. we investigated the change in the prediction module when the perception model is updated with a new data set;
  3. the 30 different test cases;
  4. entangled enhancement is observed in 23 of 30;
  5. then the comparative study for the cause.

## I. Introduction (page 1)

**D6. "call this effect" (#10, #12, edit) and italics (#11)**
- Paper: "Wang and Machida call this **effect** entangled enhancement [1], [2]."
- Sensei: strikes "effect", inserts "defecting update as"; on "entangled enhancement": "make it italic".
- After: "Wang and Machida call this defecting update as *entangled enhancement* [1], [2]."

**D7. Retraining-decision sentences (#13, instruction)**
- Paper: "Their work decides when an MLS should retrain each of its models. This decision needs the probability that a retraining causes entangled enhancement. No study has measured this probability on a real system."
- Sensei: "It is too high context. This paragraph focuses on a brief introduction of 'entangle enhancement'. Just introduce the phenomena."
- After: remove these three sentences; paragraph 1 only introduces the phenomenon.

**D8. Driving-pipeline paragraph (#14, #15, delete)**
- Paper: "A typical autonomous driving pipeline consists of three modules supported by machine learning (ML) models. The perception module finds the objects around the ego vehicle. The prediction module predicts where each object will move, and the planning module chooses the path of the ego vehicle." (struck)
- Sensei: "This is not right place to explain the AD pipeline. The first paragraph introduces entangle enhancement. Then the third paragraph follow the discussion with related work. Thus, removing the second paragraph is reasonable."
- After: paragraph deleted.

**D9. Gap sentence (#16–#20, edit)**
- Paper: "**None** of these studies **retrains the perception model of a driving pipeline and measures how often the error of the prediction and planning models increases, or which cause produces the increase.**"
- Sensei: "None" → "However, none"; strikes the rest; inserts "examined the significance of entangle enhancement in a real autonomous driving pipeline, consisting of perception, prediction, and planning modules."
- After: "However, none of these studies examined the significance of entangled enhancement in a real autonomous driving pipeline, consisting of perception, prediction, and planning modules."

**D10. Objective paragraph (#21–#25, edit and instruction)**
- Paper: "In this **report,** we empirically investigate whether an update that improves the perception model increases the error of the prediction and planning models, **and whether the increase comes from the adaptation of these models to the errors of the perception model.** Adaptation means that …"
- Sensei: "report" → "study"; strikes "and whether the increase comes from …"; inserts "in an autonomous driving pipeline. When increased errors are observed, we further investigate the impact of adaptation of the downstream models." Note: "Stop with one subject. Explain a bit details. Then, move to the next subject. (Don't connect many different subjects by 'and')"
- After: "In this study, we empirically investigate whether an update that improves the perception model increases the error of the prediction and planning models in an autonomous driving pipeline. When increased errors are observed, we further investigate the impact of adaptation of the downstream models."

**D11. Detection question (#26, question)**
- Paper: "We also examine whether entangled enhancement can be detected **without ground truth.**"
- Sensei: "by what means?"
- After: say how detection is attempted (the measures of output change), or drop the question (see D33).

**D12. Procedure sentence (#27, #28, edit and instruction)**
- Paper: "To separate the cause, we train the prediction and planning models either on the ground truth or on the output of the perception model before the update, **keep both fixed,** and update the perception model 30 times."
- Sensei: strikes "keep both fixed,"; on "and update the perception model 30 times.": "Maybe you need to explain the details of update experiments separately. What are 30 updates? Are they simple repetitions of the same update procedure? It is unclear."
- After: explain the 30 updates in their own sentences (what varies between them: amount of data, number of epochs, random seed, two starting models).

**D13. Results summary (#29–#33, edit and instruction)**
- Paper: "We observe that the prediction error increases in 23 of the 30 updates after training on the output of the perception model, **and in 5 after training on the ground truth.** As a result, we confirm that entangled enhancement occurs in the prediction model **and** comes from its adaptation to the errors of the perception model. **The planning error increases by at most 0.13 m. None of nine measures without ground truth identifies the updates with entangled enhancement.**"
- Sensei: strikes "and in 5 after training on the ground truth."; strikes "and" and inserts "."; strikes the last two sentences. On "comes from its adaptation …": "Then, you explain the meaning of 5 results. The point is explain step by step. Don't connect by 'and'"
- After: "We observe that the prediction error increases in 23 of the 30 updates after training on the output of the perception model. As a result, we confirm that entangled enhancement occurs in the prediction model." Then, as a separate step, the comparison with the 5 updates after ground-truth training and what it means for the cause.

## II. Related Work (page 1–2)

**D14. Section title (#34, edit)**
- Paper: "II. BACKGROUND AND RELATED WORK"
- Sensei: strikes "BACKGROUND AND".
- After: "II. RELATED WORK"

**D15. Opening sentence (#35, instruction)**
- Paper: "This section summarises the related work."
- Sensei: "Explain the domain of related work introduced in this paragraph (not in this section)"
- After: each paragraph opens by naming its own domain of related work.

**D16. Model paragraph (#36, instruction)**
- Paper: "The prediction and planning models of this report follow two published models. AutoBot-Ego [6] predicts the path of an object. PlanT [7] plans the path of the ego vehicle from a list of objects. Weng et al. [8] show … The prediction error increases when the input comes from a real perception model instead of the ground truth [5]. A planning model that receives the state of the ego vehicle can score well on the nuScenes driving data set while ignoring the objects [9], [10]."
- Sensei: "These are not a relevant content for related work. They are just introduction of available ML models for AD pipeline. They are not comparative target of our study. Provide proper related work and explain how your study is different from the existing studies."
- After: move the model descriptions to the setup (implementation); write related work that is comparable to this study and state the difference.

**D17. Symptom sentence (#37–#39, edit)**
- Paper: "In this report, we apply such distances to the outputs of the models as a candidate symptom, **which is** a sign of entangled enhancement **that needs no ground truth.**"
- Sensei: strikes "which is" and "that needs no ground truth."; inserts "for".
- After: "In this report, we apply such distances to the outputs of the models as a candidate symptom for a sign of entangled enhancement."

## III. Methodology (page 2)

**D18. "human driver" (#40, question)**
- Paper: "The planning error is the distance between the planned position and the position of the **human driver** at 3 s."
- Sensei: "What is your assumption here? Is groud-truth provided in data set? 'human driver' is a surprise at this point."
- After: state that the data set records the path the human driver actually drove, and that this path serves as the ground truth for planning.

**D19. "respectively" (#41, edit)**
- Paper: "We write $\Delta_2$ and $\Delta_3$ for the increase of the prediction error and of the planning error after an update."
- After: "We write $\Delta_2$ and $\Delta_3$ for the increase of the prediction error and of the planning error after an update**, respectively**."

**D20. "shows" (#42, #43, edit)**
- Paper: "An update **shows** entangled enhancement when"
- After: "An update **causes** entangled enhancement when"

**D21. Research questions and hypotheses (#44, instruction)**
- Paper: subsection "B. Research Questions and Hypotheses" (goal, RQ1–RQ3, H1–H3).
- Sensei: "I think that RQs and hypotheses are overlaped here. You may restructure this subsection with a set of research questions (RQs), instead of hypothesis." (Email: "Hypotheses are actually RQs, so you can remove them.")
- After: keep the RQs; remove H1–H3.

## IV. Experimental Setup (pages 2–3)

**D22. Structure of the section (#45, instruction)**
- Paper: section title "EXPERIMENTAL SETUP".
- Sensei: "This section is not structured well. Reconsider the order of the content and structure of subsections. Better to align the content along with individual RQs."

**D23. Data split (#46, instruction)**
- Paper: "Boston 1 (100 scenes) trains $C_1$, and Boston 2 (100 scenes) trains $C_2$ and $C_3$ …"
- Sensei: "Data split is actually explained in the next subsection. You should merge descriptions and present the data set, data set preparation, and model training in a proper order."

**D24. Workflow vs RQs (#47, #48, instruction; main point)**
- Paper: "We train $C_2$ and $C_3$ once in two training modes." (Workflow subsection)
- Sensei: "The problem of this workflow is that it does not align with the RQs. To answer RQ1, we do not need to update C2 and C3, right? The training of C2 and C3 are only necessary for answering RQ2. It is a separate experiment. You should not mix these two experiments in a sequence. I think this is the main source of confusion."
- After: two separate experiments.
  - **Experiment for RQ1:** update $C_1$ and measure the pipeline.
  - **Experiment for RQ2:** compare the two training modes of $C_2$ and $C_3$.

**D25. The 30 updates (#49, question)**
- Paper: "We then test each of **the 30 updates** in the full pipeline …"
- Sensei: "What are these 30 updates? Are they simple repetitions?"

**D26. Training-mode names (#50, question)**
- Paper: "In GT training, … In $C_1$-output training, …"
- Sensei: "Previously, you used 'iso' and 'pipe.' Why did you change the naming convention?"
- After: decide the names. Sensei expects the earlier names ("iso" for training on the ground truth, "pipe" for training on the output of $C_1$), or an explanation of the change.

**D27. "procedure for one $C_1$" (#51, question) and the Algorithm sentence (#52–#56, #58–#60, edit)**
- Paper: "Algorithm 1 gives the **procedure for one $C_1$**, where **GT marks** GT training and **out marks** $C_1$-output training."
- Sensei: on "procedure for one $C_1$": "What do you mean by this? Could not understand." Edits: "GT" → "'GT'", "marks" → "indicates" (twice), "out" → "'out'".
- After: "Algorithm 1 gives the procedure for one $C_1$, where 'GT' indicates GT training and 'out' indicates $C_1$-output training." Also explain what "for one $C_1$" means (one of the two starting models).

**D28. Seed sentence in the algorithm paragraph (#57, instruction)**
- Paper: "The two $C_1$ trained with seeds 0 and 1 have almost the same accuracy."
- Sensei: "I do not think this description is not a part of Algorithm 1. Please explain Algorithm 1, step by step."
- After: explain Algorithm 1 line by line; move the seed-variation explanation elsewhere.

**D29. Models vs update procedure (#61, instruction)**
- Paper: "We use YOLOv11n [13] as a perception model ($C_1$), which is trained for 10 passes over Boston 1 with the random seeds 0 and 1."
- Sensei: "The used models should be explained separately from the update procedure. Maybe after data set, you should explain the entire implementation of the pipeline. Then, other sections, you can explain, how these models are updated for individual RQs."
- After, the order is:
  1. data set;
  2. implementation of the whole pipeline ($C_1$, $C_2$, $C_3$);
  3. for each RQ, how the models are updated.

**D30. Fig. 1 (#62, instruction)**
- Paper: caption "Workflow of the experiment".
- Sensei: "As commented in the text, this workflow is the source of confusion. Inserting C2 and C3 updates in the middle step is not good, as they are for different purposes (RQ)."
- After: redraw Fig. 1 so that the RQ1 and RQ2 experiments are separate.

**D31. $C_3$ input sentence (#63, instruction; second time, see v0 C43)**
- Paper: "Every object is one input with the current position, the most likely path from $C_2$, the class and the confidence."
- Sensei: "This sentence is uninterpretable."

**D32. Training and Measurements (#64, instruction)**
- Paper: "We train $C_2$ and $C_3$ once with GT training and once with $C_1$-output training for each $C_1$, and the trained models stay fixed during all updates. The prediction error is computed over …"
- Sensei: "It is totally unclear why this experiment is conducted. Again, the explanation of experiments should align with RQs. Individual experiments are designed for providing answer for the corresponding RQs. In the present report, such correspondence is unclear. The entire section needs restructuring."

## V. Results and Discussion (pages 3–4)

**D33. Opening (#65, instruction)**
- Paper: "In this section, we first check that $C_2$ and $C_3$ work, because a count of updates that increase the error is meaningful only for working modules. We then report and discuss the results. Every update made $C_1$ more accurate, with $\delta_1$ between 0.048 and 0.118."
- Sensei: "Should be aligned with RQs. Each results should address the RQs and provide answers to the corresponding RQs."

**D34. Validity of the ML components (#66, instruction)**
- Paper: subsection "A. Validity of the ML Components" (baselines, Table I).
- Sensei: "I do not think that this baseline analysis is necessary because the correspondence to RQ is unclear. Provide a better context information or remove."

**D35. Prediction results (#67, question)**
- Paper: end of the first paragraph of V-B ("… lies between +0.049 and +0.284 m with a mean of +0.165 m.").
- Sensei: "Then, what? Only describing values does not provide insight. What is your interepretation of these values, with respect to RQs?"

**D36. "the first cause" (#68, question)**
- Paper: "The comparison results characterize **the first cause of entangled enhancement.**"
- Sensei: "What is the first cause of entangle enhancement? You never explained this."
- After: name the cause in the sentence itself (adaptation of the downstream models to the errors of $C_1$).

**D37. Planning results (#69, question)**
- Paper: subsection "C. Planning Module".
- Sensei: "Are these results corresponding to RQ1? If so, this is the first main result, isn't it?"
- After: if they answer RQ1, present them first, as part of the RQ1 result.

**D38. Retraining results (#70, question)**
- Paper: subsection "D. Retraining $C_2$ and $C_3$ on the Output of $C_1'$".
- Sensei: "These results are to what? For answering RQ2?"
- After: state which RQ the subsection answers.

**D39. Cost sentences (#71, delete)**
- Paper: "This retraining has a cost. A symptom of entangled enhancement is therefore useful." (struck)
- After: deleted.

**D40. Detection analysis (#72, instruction)**
- Paper: subsection "E. Symptom of Entangled Enhancement".
- Sensei: "The objective of this results are not clear by description. Maybe you can cut this analysis to make your presentation clearer. Otherwise, properly formulate RQ3, and explain the experiment design for answering RQ3, and explain the results to provide the answer to RQ3."
- After, one of two:
  - (a) cut the analysis, with RQ3;
  - (b) formulate RQ3 properly, explain its experiment design, and give the results as the answer.

## VI. Conclusion (page 4)

**D41. First sentence (#73, #74, edit)**
- Paper: "In this report, we **updated the perception module 30 times.**"
- After: "In this report, we investigated the entangled enhancement in a real autonomous driving pipeline."

**D42. Summary (#75, instruction)**
- Paper: "We observe that the prediction module became worse more often and by more after $C_1$-output training than after GT training. This result supports H1 and H2 for the prediction model and points to adaptation to the errors of the perception model before the update as the cause. …"
- Sensei: "Provide a proper summary. Like abstract, what pipeline you considered. (dataset, models, update C1, etc.) Then, explain the main results in the right order. (answer to RQ1, RQ2)"

---

## What this round asks for, in short

**A. Restructure around the RQs (the main request).**
- Email: "the key is a set of RQs"; remove the hypotheses (D21).
- Section IV in this order:
  1. data set and its preparation;
  2. implementation of the whole pipeline;
  3. one experiment per RQ (D22–D24, D29, D32).
- The **RQ1 experiment** updates $C_1$ only. The **RQ2 experiment** is a separate comparison of the two training modes of $C_2$ and $C_3$. Mixing them in one workflow "is the main source of confusion" (D24, D30).
- Section V gives one result per RQ, with an interpretation and an answer (D33, D35, D37, D38).
- RQ3: formulate it properly or cut it (D11, D40).
- The baseline check: give it context or remove it (D34).

**B. Explain what is still unclear.**
- What the 30 updates are (D12, D25).
- What "procedure for one $C_1$" means (D27).
- What the "first cause" is (D36).
- Where "human driver" comes from (D18).
- The $C_3$ input sentence (D31).
- Explain Algorithm 1 step by step (D28).

**C. Shorter, focused Introduction and Related Work.**
- Introduction paragraph 1 only introduces the phenomenon (D7).
- Delete the driving-pipeline paragraph (D8).
- Related work must be comparable studies, not the models used (D15, D16).
- Remove "Background" from the section title (D14).

**D. Abstract and Conclusion follow the experiments in order.**
- Pipeline and data set → updates of the perception model → 30 test cases → entangled enhancement in 23 of 30 → comparative study for the cause (D5, D13, D42).

**E. Word-level edits** (his own wording):
- "study" not "report" in the abstract and Introduction (D4, D10);
- "real" (D3, D4, D9, D10, D41);
- "However, none … examined the significance of entangled enhancement in a real autonomous driving pipeline" (D9);
- "defecting update as *entangled enhancement*", in italics (D6);
- "respectively" (D19);
- "causes entangled enhancement" (D20);
- "'GT' indicates …, 'out' indicates …" (D27);
- "performance" removed (D2);
- "Related Work" (D14).
- One sentence per subject; do not join different subjects with "and" (D10, D13).

**F. Names.** He asks why "iso" and "pipe" became "GT training" and "$C_1$-output training" (D26). The training-mode names need a decision.

**G. Priority.** The report comes before the presentation (email).
