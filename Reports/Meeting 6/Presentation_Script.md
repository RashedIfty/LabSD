# Meeting 6: presentation script

Speak it, don't read it. Each part is about 1 minute.

---

## 0. Opening (answer to your two emails)

"Thank you for the two emails. I revised the report for all of your comments. I'll go through it section by section, and I'll say which comment each change answers.

First, the format. The report now follows the normal paper format: Abstract, Introduction, Background, Hypotheses, Experimental Design, Conclusion. I removed all the meeting context, the sentences about the review comments of the base papers, and the presentation-style section titles.

For the writing, I removed every 'so', I replaced unclear pronouns with the actual noun, and I applied your corrections: 'ego vehicle', 'this study', 'increase', and 'ML models' or 'ML components' instead of 'learned models'. I also collected all of your writing rules from the paper reviews into one master file, and I checked this report against it."

---

## 1. Introduction (the problem and the gap)

"The starting point is one number in your models, c1: the probability that retraining the upstream component leaves the downstream component working. It was assumed to be 0.75 and has never been measured.

In our earlier experiments, retraining the perception module made it more accurate, but the planning error increased in 6 of 15 updates. However, the prediction and planning modules were fixed rules. Fixed rules cannot adapt to the mistakes of the perception module, and that adaptation is exactly the cause the theory describes. The earlier result was real, but it did not test the theory.

**New in this report:** a second question, following your email. In a deployed system there is no ground truth, and accuracy cannot be measured. Can we find a symptom of entangled enhancement without ground truth?"

---

## 2. Background (only what the reader needs)

"This section defines every term before it is used: the pipeline, the ego vehicle, the ground truth, an update, the old and new C1, and the prediction and planning errors.

It also gives the two possible causes of harm. First, the downstream modules adapted to the old mistakes of C1. Second, C1 improved on average but became worse on the cases that matter downstream.

**New:** a short paragraph on measuring a change in a distribution without labels, citing Rabanser et al., with the two measures we use: Jensen–Shannon divergence and Wasserstein distance."
 
---

## 3. Hypotheses (the main change)

**H1 (your comment: 'the hypothesis is doubtful')**

"You were right that the planning error does not always increase. H1 now says only what can happen: an update of the perception module *can* reduce the performance of the prediction and planning modules, even when the perception module itself improves. It does not claim that every update does this. It claims that a clear number of updates do, more than random variation alone, and in several pipelines. Figure 1 shows one such update."

**H2 (your comment: 'the fundamental question is how to find the symptom')**

"This is the new H2. The change in the output distribution of C1 and C2 after an update, measured on the same images without ground truth, is larger in harmful updates than in harmless ones. The accuracy gain of C1 alone cannot tell them apart.

The clue towards a solution: before using a retrained C1, measure how much the outputs of C1 and C2 change on unlabelled images. If the change is large, retrain C2 and C3 as well, instead of updating C1 alone."

**Old H2 (your comment: 'ok for an empirical study, but no clue for a solution')**

"I agreed. The old H2 is no longer a hypothesis. The frequency of harmful updates, which gives a measured 1 − c1, is now only a by-product of the same experiment."

---

## 4. Experimental design

**4.1 Dataset (your comment: 'good attempt')**

"The dataset is built and verified. It went from 559 scenes and 45 GB to 350 scenes and 4.6 GB, with full-size images, and the test scenes are the same as before. Table 2 compares the two datasets."

**4.2 ML models and pipelines (your comment: 'test diverse pipelines')**

"Instead of one model with a fallback, there are now two ML models for each downstream module, combined into four pipelines, P1 to P4 in Table 3. For prediction: AutoBot-Ego and Social-LSTM. For planning: PlanT and AD-MLP. Every module is an ML component, including the step that matches objects across images. Before any update is run, I check that each planning model really uses the objects."

**4.3 Three ways of training C2 and C3**

"C2 and C3 learn either from the ground truth, from the output of the old C1, or from the output of the new C1. Comparing the three tells us which of the two causes produced the harm. Table 1 shows what each case predicts."

**4.4 Measures of output change (new, answers the symptom question)**

"Table 4 lists the metrics, and none of them needs ground truth. For C1: the number of objects per image, the mix of object types, confidence scores, object sizes and positions, and the fraction of objects that only one version finds. For C2: the predicted movements, and how far the predicted position of the same object shifts. H2 holds if these measures separate harmful from harmless updates better than the accuracy of C1 does."

**4.5 Checks on the measurement**

"Two checks. The unchanged modules must give exactly the same result on ground truth input. And an increase counts only if it is larger than the increase from retraining the old C1 with only a different random seed."

---

## 5. Conclusion and next step

"To summarise: H1 is now a safe claim that the effect can happen, H2 addresses your fundamental question about a symptom without ground truth, and four pipelines test whether the effect is general. The dataset is ready.

The next step is the first run: retrain the old C1 with two random seeds on the reduced dataset, to measure random variation and confirm that 350 scenes are enough."

---

## Points to raise for discussion

1. "The report gives the clue towards a solution, but not yet the method itself, for example the threshold for a 'large' change. Should I add it now, or after the first results?"
2. "Standard metrics I could add: the Kolmogorov–Smirnov test and Maximum Mean Discrepancy. Our 'only one version finds' measure is known as prediction churn. Would you like me to use those standard names?"
3. "Our earlier interface-drift score did not separate harmful updates (AUROC 0.48), but it was tested with fixed-rule modules. Should I mention it in the report?"
