# Sensei's comments on CS_seminar_IFTY (v0), against the paper text

Sources:
- **Email 1** (Fumio Machida, 12 October 2026, 17:15, "RE: CS seminar report"), with the first-page comments.
- **Email 2** (follow-up), with the full commented report.
- **`CS_seminar_IFTY-v0-comment2.pdf`**: 116 marks on all four pages (48 highlights, 26 strike-outs, 22 insertions, 17 notes, 3 lines). It contains the first-page comments of Email 1 as well.

`CS_seminar_IFTY.tex` has not changed since the version sensei marked (last commit 1bfc7ff). The line numbers below therefore point to the exact source he commented on.

How each comment is shown:
- **Paper:** the exact text of the paper, with the line number in `CS_seminar_IFTY.tex`.
- **Sensei:** the mark (struck words, inserted words, or his note), quoted as written.
- **After:** for a wording edit, the sentence with his edit applied. For an instruction or question, what has to change.
- `#n`: the mark number in reading order (page, left column, right column, top to bottom).

---

## Emails

**E1.** "We need to prepare an understandable report. This is the primary goal you should focus on this week. The current report is not understandable. It is not an academic report. Let us improve step by step. The initial comments on the first page are attached."

**E2.** "Further comments are included in the attached report. I find that you seldom use 'we' as the subject of sentences, which might be one of the reasons why your report is hard to read. You can use 'we' as the subject. Clarify what you did (introduction, experiments, conclusion)."
Examples: "In this report, we conducted …" · "We retrained …" · "We used 350 scenes of nuScenes dataset for the …" · "We split the data …" · "We observe that …" · "Next, we examine the hypothesis …" · "As a result, we confirm that …"

---

## Title and abstract (page 1)

**C1. Title (#1, edit)**
- Paper (l. 22): "An Empirical Analysis of Entangled Enhancement in an Autonomous Driving Pipeline **of ML Components**"
- Sensei: strikes "of ML Components".
- After: "An Empirical Analysis of Entangled Enhancement in an Autonomous Driving Pipeline"

**C2. Abstract, sentence 1 (#2–#5, edit)**
- Paper (l. 40–41): "Entangled enhancement is a retraining that improves **one component of a machine learning system (MLS)** but fails to improve the whole system."
- Sensei: strikes "one component of a machine learning system (MLS)". Inserts "defective" before "retraining", "of machine learning (ML) model" after "retraining", and "the updated model performance" in place of the struck words.
- After: "Entangled enhancement is a **defective** retraining **of machine learning (ML) model** that improves **the updated model performance** but fails to improve the whole system."

**C3. Abstract, before introducing the study (#6, instruction)**
- Paper (l. 40–41): the abstract starts with the definition and goes straight to "This paper measures …".
- Sensei: "Before introducing this study, it is necessary to clarify the limitation of the existing research, which you address in your work."
- After: add the limitation of existing research (the gap) before the sentence about this study.

**C4–C6. Abstract, sentence 2 (#7–#13, edit)**
- Paper (l. 41): "This **paper measures** entangled enhancement in **a** driving pipeline **whose** perception, prediction and planning modules **are machine learning (ML) components.**"
- Sensei: "paper measures" → "report empirically investigates". "a" → "an autonomous". "whose" → "consisting of". Strikes "are machine learning (ML) components."
- After: "This **report empirically investigates** entangled enhancement in **an autonomous** driving pipeline **consisting of** perception, prediction and planning modules."

**C7. Abstract, rest (#14, instruction)**
- Paper (l. 41–45): "The perception module is improved by 30 updates, while the prediction and planning modules stay fixed after training on the ground truth or on the output of the old perception module. When the prediction module was trained on the output of the old perception module, the prediction error increased in 23 of the 30 updates, in 11 by more than random variation and in 3 with statistical significance. When the prediction module was trained on the ground truth, the counts were 5, 0 and 0. The planning error changed by at most 0.13 m. No measure of output change that needs no ground truth predicted which updates harmed the prediction module."
- Sensei: "Need complete rewrite. Clarify objective. Explain the evaluation procedure. Then, explain the results (which should be along with the objective above mentioned). For example, if the main objective is to confirm the existence of entangle enhancement in AV pipeline, the results give answers to that question."
- After: rewrite the abstract as gap → objective → evaluation procedure → results that answer the objective.

---

## Introduction (page 1)

**C8. MLS spelled out (#15, #16, edit)**
- Paper (l. 56): "An **MLS** is often built as a pipeline, …"
- Sensei: "MLS" → "Machine Learning System (MLS)".
- After: "A **Machine Learning System (MLS)** is often built as a pipeline, …"

**C9. "in a line" (#17, #18, edit)**
- Paper (l. 56–57): "… which is a set of ML components connected **in a line,** where each component uses the output of the preceding component."
- Sensei: strikes "in a line,". Inserts "sequentially" before "connected".
- After: "… which is a set of ML components **sequentially** connected, where each component uses the output of the preceding component."

**C10. "line" (#19, edit)**
- Paper (l. 58): "A component that comes earlier in the **line** is upstream of a component that comes later, …"
- Sensei: "line" → "sequence".
- After: "A component that comes earlier in the **sequence** is upstream of …"

**C11. Probability of harm (#20, question)**
- Paper (l. 60): "The models need a probability of harm that has not been measured on a real system."
- Sensei: "Not clear."
- After: state plainly which probability the models of Wang and Machida need, and that nobody has measured it in a driving pipeline. Remove "harm".

**C12–C14. Driving pipeline (#21–#26, edit)**
- Paper (l. 62–63): "A driving pipeline **has** three **components.** The perception module finds the objects around the ego vehicle**, which is the vehicle that the pipeline drives.**"
- Sensei: inserts "typical autonomous" before "driving". "has" → "consists of". "components." → "modules supported by machine learning (ML) models". Strikes ", which is the vehicle that the pipeline drives."
- After: "A **typical autonomous** driving pipeline **consists of** three **modules supported by machine learning (ML) models**. The perception module finds the objects around the ego vehicle."

**C15. Earlier experiment (#27, instruction)**
- Paper (l. 65–67): "An earlier experiment in this study retrained the perception module of such a pipeline 15 times. In 6 of the 15 updates, the planned path moved further from the path of the human driver. The prediction module and the planning module in that pipeline were fixed rules. Fixed rules cannot adapt to the mistakes of the perception module. That experiment could not test the adaptation that the theory of entangled enhancement describes."
- Sensei: "This is your personal context which is unnecessary for the report. Instead, you should clarify the existing studies and their limitations."
- After: delete the paragraph. Describe existing studies and their limitations instead.

**C16. Difference from earlier experiment (#28, instruction)**
- Paper (l. 69): "**This paper replaces both fixed rules with ML components**, trained once on the ground truth, the object positions labelled by hand, and once on the output of the old perception module. Only the second way allows adaptation to the old mistakes. The comparison of the two ways therefore shows whether a harm comes from that adaptation."
- Sensei: "Instead of the difference from your previous experiments, you should clarify the difference from the existing studies."
- After: state how this study differs from existing studies, not from the earlier experiment.

**C17. Contributions (#29, #30, instruction)**
- Paper (l. 71–79): "This paper makes three contributions. 1) A driving pipeline of three ML components, trained on a reduced version of the nuScenes dataset [3] and checked against simple baselines. 2) A series of 30 updates of the perception module, with the number of harmful updates counted at three levels of evidence. 3) A test of nine measures of output change as a possible symptom of entangled enhancement, which is a sign of harm that needs no ground truth." (all struck)
- Sensei: "Rather than the contributions, the following is more critical content you need to write. Explain the limitations of existing studies. (Gap) Then, clarify the objective of your study. For the objective, clarify what you attempted in this report. Then, explain the summary evaluation results."
- After: replace the contribution list with gap → objective → what we attempted → summary of the results.

---

## Section II, Background and Related Work (page 1)

**C18. Section opening (#31, #32, edit)**
- Paper (l. 85): "This section summarises the **earlier work that the experiment builds on.**"
- Sensei: → "related work".
- After: "This section summarises the **related work**."

**C19. "Harm" everywhere (#33, instruction)**
- Paper: "harm", "harmful", "harmless", "harmed" (l. 45, 60, 69, 77, 78, 89, 263, 321, 350, 352, 356, 361, 363, 372, 373).
- Sensei: "Do not use 'harm,' 'harmful' and 'harmless,' whose meanings are unclear and subjective expressions."
- After: remove every form of "harm". Name the measured quantity, for example "the prediction error increases".

**C20–C22. The two causes (#34–#38, edit)**
- Paper (l. 88–93): "**A harm of this kind can have two causes. In the first cause,** the downstream components adapted during training to the mistakes of the old upstream component and are disturbed when those mistakes change. **In the second cause,** the upstream component improves on average but becomes worse on the cases that matter downstream."
- Sensei: strikes "A harm of this kind can have two causes." "In the first cause," → "First,". "In the second cause," → "Second,".
- After: "**First,** the downstream components adapted during training to the mistakes of the old upstream component and are disturbed when those mistakes change. **Second,** the upstream component improves on average but becomes worse on the cases that matter downstream."

**C23. "paper" → "report" (#39, #40, edit)**
- Paper (l. 105): "This **paper** applies such distances to the outputs of the modules as a candidate symptom."
- After: "This **report** applies such distances …" (the same applies to every "this paper").

---

## Section III, Methodology (page 2)

**C24. Define the pipeline and C1–C3 again (#41, #42, instruction)**
- Paper (l. 114–117): "The perception module $C_1$ finds the objects in the image of the front camera. The prediction module $C_2$ predicts the path of each object over the next 3 s, and the planning module $C_3$ chooses the path of the ego vehicle over the same 3 s."
- Sensei: "Explain that this report focuses on an AV pipeline consisting of perception, prediction, and planning modules, explicitly again. Then, define C1, C2, and C3 explicitly."
- After: first state that the report focuses on this pipeline. Then define $C_1$, $C_2$ and $C_3$ one by one.

**C25. "from a new city" (#43, question)**
- Paper (l. 117): "An update is one retraining of $C_1$ on data **from a new city**."
- Sensei: "The context is not introduced yet."
- After: explain the cities (the data split) before using "new city", or drop it here.

**C26. Old/new $C_1$ (#44, instruction)**
- Paper (l. 117–118): "We call $C_1$ before an update the old $C_1$ and $C_1$ after the update the new $C_1$."
- Sensei: "Why do not you use the notations introduced in the previous report. C1' as an updated version. Which looks more formal. old/new is just relative and hence not good naming convention."
- After: use $C_1$ and $C_1'$ everywhere.

**C27, C28. Before the hypotheses (#45, #46, question and instruction)**
- Paper (l. 133–134): the subsection "Hypotheses" starts directly with "H1 (existence): …".
- Sensei: "What is the goal of this study? Why do you describe these hypothesis. Note that they are not obvious, even though you thought they are obvious." And: "Describe something before jumping into the set of hypothesis."
- After: state the goal of the study and why each hypothesis follows from it before listing them.

**C29. H1 label (#47, instruction)**
- Paper (l. 134): "H1 **(existence)**:"
- Sensei: "The label is too short, and not understandable."

**C30. H1 tense (#48, #49, edit)**
- Paper (l. 136–138): "The increase is larger when $C_2$ and $C_3$ **were** trained on the output of the old $C_1$ than when $C_2$ and $C_3$ **were** trained on the ground truth."
- After: "… when $C_2$ and $C_3$ **are** trained on the output of the old $C_1$ than when $C_2$ and $C_3$ **are** trained on the ground truth."

**C31. Reason for H1 (#50, question)**
- Paper (l. 136–138): the second sentence of H1 (above).
- Sensei: "Need more explanation. Why do you think so."
- After: explain why training on the output of $C_1$ should make the increase larger.

**C32. H2 label (#51, instruction)**
- Paper (l. 140): "H2 **(symptom)**:"
- Sensei: "The label is too short, and not understandable."

**C33. H2 statement (#52, instruction)**
- Paper (l. 140–142): "The change in the output of $C_1$ and $C_2$ after an update, measured without ground truth, is larger for the updates that meet condition (1) than for the other updates. The gain $\delta_1$ alone does not separate the two groups."
- Sensei: "I could not understand. Consider rephrase this with simpler and precise words."

**C34. Workflow placement (#53, question)**
- Paper (l. 144–150): "Fig. 1 shows the six stages of the experiment. The data are split into four parts. Boston 1 and Boston 2 train the modules, Singapore 1 provides the data for the updates, and Singapore 2 is the test set. The old $C_1$ is trained with two random seeds, … $C_2$ and $C_3$ are trained once in two ways. Isolated training uses the ground truth, and old-pipeline training uses the output of the old $C_1$. Each of the 30 updates is then tested in the full pipeline with $C_2$ and $C_3$ kept fixed."
- Sensei: "Not sure why the workflow is presented here, separately from the experimental setup. Is it a part of experimental setup, isn't it? It is strange to explain with Singapore and Boston before introducing the details of data set."
- After: move the workflow into the experimental setup, after the data set.

**C35. Training names (#54–#58, instruction)**
- Paper (l. 149, 183, Algorithm 1): "isolated training", "old-pipeline training", "iso", "op" (highlighted, lines beside "D. Update Procedure").
- Sensei: "These terms are not properly defined before using. 'old-pipline training' does not make sense. Consider a proper alternative term."
- After: new names, defined before first use (decided: **GT training** and **$C_1$-output training**).

---

## Section IV, Experimental Setup (page 2)

**C36. Section opening (#59, #60, edit)**
- Paper (l. 219–220): "This section **gives the data, the updates of $C_1$, the ML models for $C_2$ and $C_3$, and the measurements.**"
- After: "This section **presents the data set and the ML models used for experiments**."

**C37. Hardware (#61, instruction)**
- Paper (l. 220): "All models were trained on one **Tesla T4 graphics card**."
- Sensei: "Not only GPU, but also other specs need to be clarified."

**C38. Order of sections (#62, instruction)**
- Paper (l. 223): "**The experiment uses 350 of the 850 scenes of nuScenes [3],** …"
- Sensei: "The previous section already introduces the details of training process with these data set. The presenter order is strange. The organization needs to be reconsidered."

**C39. C1 sentence (#63–#65, edit)**
- Paper (l. 232–233): "**$C_1$ is** YOLOv11n [13], trained for 10 passes over Boston 1 with the random seeds 0 and 1."
- After: "**We use** YOLOv11n [13] **as a perception model ($C_1$), which is** trained for 10 passes over Boston 1 with the random seeds 0 and 1."

**C40. "old" (#66, #67, mark)**
- Paper (l. 233–234): "The two **old** $C_1$ reach …", "An update retrains an **old** $C_1$ …". See C26.

**C41. C2 sentence (#68, #69, edit)**
- Paper (l. 239): "**$C_2$ follows** AutoBot-Ego [6]."
- After: "AutoBot-Ego [6] **is employed as a prediction model ($C_2$)**."

**C42. C3 sentence (#70, #72, edit)**
- Paper (l. 241): "**$C_3$ follows** PlanT [10]."
- After: "**For the planning model ($C_3$), we employ** PlanT [10]."

**C43. C3 input sentence (#71, instruction)**
- Paper (l. 241): "Every object is one input with the current position, the most likely path from $C_2$, the class and the confidence."
- Sensei: "Can you make it more direct sentence. The sentence is not interpretable."

---

## Sections IV and V (page 3)

**C44. Object-use check (#73, instruction)**
- Paper (l. 244–248): "An input of the ego path can let a planning module ignore the objects [11], and the experiment checks whether $C_3$ uses the objects. Removing the objects from the input of $C_3$ moves the planned position at 3 s by 0.73 to 1.65 m on average, which is above the threshold of 0.5 m fixed before the experiment."
- Sensei: "Moderate the sentences. Hard to follow."

**C45. Collision rate (#74, instruction)**
- Paper (l. 252–253): "**The collision rate** is the fraction of test samples in which the planned path overlaps an object." (also l. 269–270)
- Sensei: "If it is not presented in the results, it does not necessary to mention. Mind the consistency of the report. Write a complete report understandable."

**C46. Random variation (#75, instruction)**
- Paper (l. 255–256): "The random variation is $\tau_2 = 0.112$ m and $\tau_3 = 0.018$ m."
- Sensei: "I believe the meaning of 'random variation' is not properly presented."

**C47. "harmful" in the Section V opening (#76–#78, question)**
- Paper (l. 263): "This section first checks that $C_2$ and $C_3$ work, because a count of **harmful updates** is meaningful only for working modules, and then reports and discusses the results."
- Sensei: "Avoid using 'harmful'. Is it different from entangle enhancement?"

**C48. Structure of the results (#79, instruction)**
- Paper (l. 263–264, Section V as a whole): "… Every update made $C_1$ more accurate, with $\delta_1$ between 0.048 and 0.118."
- Sensei: "In general, evaluation results are not structured well. It looks like an arbitrary experimental reports. It should be along with your research questions. Think about what to present and why?"

**C49. Validity of the ML components (#80, question)**
- Paper (l. 266–270): "Table I compares $C_2$ and $C_3$ with simple baselines on the test set. $C_2$ beats the baseline in which every object stays still. The variant of $C_3$ without the ego path does not beat the baseline that drives the average Boston path for each command. The experiment uses the variant with the ego path, which has a collision rate of 0.05 to 0.08 against 0.18 for the baseline."
- Sensei: "The meaning of these evaluations are unclear. Why do we need to compare with baselines? Why do C2 and C3 need to beat baselines? Why the validity of ML components are confirmed only by this evaluation?"

**C50, C51. Δ2 and "both ways of training" (#81–#83, instruction and edit)**
- Paper (l. 292): "Fig. 2 shows **$\Delta_2$** for every update **and both ways of training.**"
- Sensei: "Although it is explained before, it is kind to explain what this symbol mean." And "and both ways of training." → "in two training modes."
- After: "Fig. 2 shows $\Delta_2$, **the increase of the prediction error,** for every update **in two training modes**."

**C52. "points to" (#84, #85, edit)**
- Paper (l. 321): "The comparison **points to** the first cause of entangled enhancement."
- Sensei: "points to" → "results characterize".
- After: "The comparison **results characterize** the first cause of entangled enhancement."

**C53. "the same new $C_1$" (#86, edit)**
- Paper (l. 321): "Both versions of $C_2$ receive the same output of **the same new $C_1$**, …"
- Sensei: "C1' or the updated C1".
- After: "Both versions of $C_2$ receive the same output of **$C_1'$** …"

**C54. Where to look (#87, question)**
- Paper (l. 321): "… and only the version that learned from the old $C_1$ becomes worse."
- Sensei: "Where should I look at?"
- After: point to the figure or table that shows this.

**C55. The seed swap (#88, question)**
- Paper (l. 321): "The $C_2$ trained on the old $C_1$ with seed 0 also becomes worse by 0.112 m when the old $C_1$ is replaced by the other old $C_1$, although the two old $C_1$ differ by only 0.009 in mAP50."
- Sensei: "Not sure where are you talking about."

**C56. Marks (#89–#91)**
- Paper (l. 321): "**0.098 m**", "No update from seed 1 **harms** this $C_2$ …", "The **harm** after an update …". See C19.

**C57. Takeaway of V-B (#92, question)**
- Paper (l. 321, end of V-B): "The harm after an update is larger for the old $C_1$ with the stronger adaptation."
- Sensei: "The conclusion is unclear. What is takeaway from Fig2 and Table II?"

---

## Sections V and VI (page 4)

**C58. Purpose of V-C (#93, question)**
- Paper (l. 342–343): "C. Planning Module. The planning error changed little. …"
- Sensei: "What would you like to confirm?"

**C59. Ego path (#94, question)**
- Paper (l. 347): "$\Delta_3$ is small because **$C_3$ plans mainly from the ego path**."
- Sensei: "could not understand why?"

**C60. Second cause (#95, question)**
- Paper (l. 347): "The more frequent increase after isolated training points to the second cause."
- Sensei: "?"

**C61. Conclusion of V-C (#96, question)**
- Paper (l. 347, end of V-C): "… can therefore affect this $C_3$ more than the $C_3$ trained on the output of the old $C_1$."
- Sensei: "Then, what is your conclusion with respect to original motivation to test."

**C62. "New-pipeline training" (#97–#99, instruction)**
- Paper (l. 349–350): heading "D. **New-Pipeline Training**", "**New-pipeline training** tests whether retraining $C_2$ and $C_3$ together with $C_1$ removes the harm."
- Sensei: "Consider a proper term."

**C63. "removes the harm" (#100, instruction)**
- Paper (l. 350): "… tests whether retraining $C_2$ and $C_3$ together with $C_1$ **removes the harm**."
- Sensei: "Write it more precisely. the harm is not understandable."

**C64. The nine measures (#101, question)**
- Paper (l. 355): "Nine measures of output change need no ground truth, seven on the output of $C_1$ and two on the output of $C_2$."
- Sensei: "What are you talking about?"

**C65. Rank correlation (#102, question)**
- Paper (l. 355): "The **rank correlation** with $\Delta_2$ is close to 1 when a larger measure always comes with a larger $\Delta_2$."
- Sensei: "Why do you suddenly introduce rank correlation? I think most of the results, you omit the explanation of objective along with the main goal."

**C66. Marks (#103, #104)**
- Paper (l. 355–357): "The AUROC is the probability that a **harmful update** ($\Delta_2 > 0$) has a larger measure than a **harmless update**." See C19.

**C67. "and" (#105, instruction)**
- Paper (l. 361): "No measure separates harmful and harmless updates consistently, **and** H2 is not supported."
- Sensei: "How these two sentences are connected. Don't use 'and' like this way. Mind the logical sentence."

**C68. "harm" in the discussion (#106–#108, instruction)**
- Paper (l. 363): "The rank correlation of $\delta_1$ shows that the accuracy gain of $C_1$ does not indicate the **harm** to $C_2$. … The **harm** depends on how strongly $C_2$ adapted to the old $C_1$."
- Sensei: "Since the meaning of 'harm' is inappropriate, the whole discussion is unclear."

**C69. "better symptom" (#109, instruction)**
- Paper (l. 363): "The **random variation in Section V-B suggests a better symptom**."
- Sensei: "Neither 'random variation' or 'better symptom' is understandable."

**C70. Mark (#110)**
- Paper (l. 363): "… the old $C_1$ whose updates **harmed** $C_2$." See C19.

**C71. Conclusion of V-E (#111, question)**
- Paper (l. 363, end of V-E): "A measure of how strongly $C_2$ reacts to any change of the output of $C_1$ can describe the adaptation of $C_2$ before an update is made."
- Sensei: "Again, so what? What is the conclusion of this subsection."

**C72. Conclusion, sentence 1 (#112, instruction)**
- Paper (l. 369): "In 30 updates of the perception module, the prediction module became worse more often and by more after old-pipeline training than after isolated training."
- Sensei: "Make it more simple and straight forward. The sentence is not straight forward."

**C73, C74. Conclusion, sentence 2 (#113, #114, question and instruction)**
- Paper (l. 369–371): "This result supports H1 for the prediction module **and** points to **adaptation to the mistakes** of the old perception module as the cause."
- Sensei: "What do you mean 'adaption to the mistakes'?" And on "and": "This 'and' is not good. Do not connect different things that are not in parallel."

**C75. Conclusion, sentence 4 (#115, instruction)**
- Paper (l. 372): "No measure of output change that needs no ground truth predicted the harm, **and** H2 is not supported."
- Sensei: "the same issue here."

**C76. "Work in progress" (#116, instruction)**
- Paper (l. 372–374): "**Work in progress** develops a method for the early detection of entangled enhancement, which flags a harmful update of the perception module before the update is used and before ground truth for the new data exists."
- Sensei: "We expect that the sentence like 'In this report, we ...'"

---

## Rules that follow for the whole report

Sensei treats each comment as an example, not a complete list.

1. **Story** (E1, C3, C7, C17, C27, C48, C57, C61, C65, C71): limitation of existing studies, then the objective, then what we did, then results that answer the objective. Every result subsection states what it tests and ends with a conclusion.
2. **"We" as the subject** (E2, C76).
3. **No "harm", "harmful", "harmless", "harmed"** (C19, C47, C56, C63, C66, C68, C70).
4. **Notation $C_1$ and $C_1'$**, never old and new (C26, C40, C53).
5. **Proper training names, defined before use** (C35, C62).
6. **Define before use** (C25, C34, C46, C50, C64, C65, C69, C73): new city, workflow, random variation, $\Delta_2$, the nine measures, rank correlation, adaptation.
7. **No personal context** (C15, C16): compare with existing studies.
8. **Order** (C34, C38): data set first; the workflow belongs to the experimental setup.
9. **Direct, simple sentences** (C43, C44, C67, C72, C74, C75). Do not join non-parallel things with "and".
10. **Wording:** "report" not "paper" (C4, C23); "sequentially" (C9); present tense (C30); "First," and "Second," (C20–C22).
11. **Consistency** (C45): no measure, such as the collision rate, that the results do not use.
12. **Hardware beyond the GPU** (C37).
