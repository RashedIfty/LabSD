# Speaker Script: An Empirical Analysis of Entangled Enhancement in an Autonomous Driving Pipeline

Ifty · 0ALD515 Research in Computer Science B · University of Tsukuba · 22 October 2026

Main talk: slides 1 to 18, about 1,650 spoken words, about 15 minutes at a slow 110 words per minute and 13 minutes at 130. The appendix slides are shown only when a question needs them.

How to read this script:

- It is written to be spoken in a semi-formal, academic tone: short sentences, plain words, one idea at a time, each slide linked to the one before.
- *(Words in italics and brackets)* are cues for where to point on the slide. Do not read them aloud.
- Say the symbols like this: C1 = "C-one", C1′ = "C-one prime", Δ2 = "delta-two", iso = "eye-so".
- Wording follows SENSEI_WRITING_MASTER.md: no "harm", "old", "so", "our car" or "detector" for a model; every term is defined the first time it is spoken.
- The talk follows the revised report (sensei's second review): two research questions, one experiment and one result for each.

The deck CS_presentation_IFTY.pptx is edited by hand (see MANUAL_EDITS.md). Edit this script directly; do not regenerate the deck or the script with build_slides.py.

## Slide 1: Cover  (about 22 s)

Good afternoon, everyone. My name is Ifty, from the Dependable AI Systems Laboratory, supervised by Professor Machida. In this presentation, I report an empirical analysis of entangled enhancement in an autonomous driving pipeline. The question is what happens to the whole system when one of its models improves.

## Slide 2: Outline  (about 12 s)

The presentation has five parts. I first introduce the problem and the existing studies. I then present the research questions, the setup, the results, and the conclusion.

## Slide 3: Background: Entangled Enhancement  (about 75 s)

I begin with the system that we study. An autonomous vehicle uses a chain of machine learning models. We call this chain a pipeline. *(point to the three boxes)* The first model, C-one, takes the camera image and finds the objects, such as vehicles and pedestrians. The second model, C-two, predicts where each object will move. The third model, C-three, plans the path of the autonomous vehicle.

Suppose that we retrain C-one and it becomes more accurate. We call the retrained model C-one prime. We would expect the whole vehicle to drive better. *(point to the definition box)* Wang and Machida showed that this is not always true. They call this defective retraining entangled enhancement. The updated model improves, but the whole system does not.

*(point to the bottom box)* This study examines one possible cause. C-two and C-three learned from the output of C-one, including its errors. They adapted to those errors. An analogy helps here. Consider a driver whose navigator always announces a turn slightly too late. The driver learns to turn slightly early. If the navigator becomes punctual, the driver now turns too early.

## Slide 4: Motivation  (about 45 s)

Why is this problem important? *(point to item 1)* Updates are routine. When a vehicle starts driving in a new city, we retrain C-one on data from that city. *(item 2)* C-two and C-three are trained on the output of C-one. They learn to work with the errors of C-one. *(item 3)* As we have just seen, a better C-one does not guarantee a better pipeline. *(item 4)* Existing studies describe this risk, but they do not examine it in a real driving pipeline.

*(point to the goal box)* Our primary goal is to find out whether a better C-one makes C-two and C-three worse in a real pipeline, and whether their adaptation to the errors of C-one is the cause.

## Slide 5: Existing Studies and Their Limitation  (about 40 s)

Several existing studies address parts of this problem. *(go down the table)* Sculley and colleagues warned that changing one model affects every model that uses its output. Wu and colleagues showed that improving one model can make the whole system worse. Wang and Machida built models that decide when to retrain. Those models need the probability of entangled enhancement as an input. Xu and colleagues showed that the prediction error increases when the input comes from a real perception model instead of the ground truth. Ground truth means the objects labelled by hand.

*(point to the gap box)* However, none of these studies examined entangled enhancement in a real driving pipeline of perception, prediction and planning models.

## Slide 6: Objective and Research Questions  (about 45 s)

This gap leads to our objective. We investigate whether an update that improves C-one increases the error of C-two and C-three. When the error increases, we further investigate the cause. We ask two research questions.

*(point to RQ1)* The first question: does an update that improves C-one increase the error of C-two or C-three? *(point to its experiment box)* To answer it, we update C-one 30 times, and keep C-two and C-three fixed.

*(RQ2)* The second question: does the increase come from the adaptation of C-two and C-three to the errors of C-one? *(its experiment box)* To answer it, we compare two versions of C-two and C-three. One version learns from the ground truth. The other learns from the output of C-one.

## Slide 7: Target Pipeline and Notation  (about 50 s)

Here is our pipeline in more detail. *(point to each box)* For C-one, we use YOLO, a common object detector applied to the camera image. For C-two, we use a prediction model in the style of AutoBot-Ego. It predicts six possible paths, three seconds long, for every object. For C-three, we use a planning model in the style of PlanT. It receives the objects, their predicted paths, the driving command, and the recent path of the autonomous vehicle.

*(point to the bottom row)* We measure three numbers. Delta-one is how much C-one improved. Delta-two is how much the prediction error increased. Delta-three is how much the planning error increased. The data set records the path that a human driver drove, and we use this path as the correct answer for planning. When C-one improves, but delta-two or delta-three is above zero, the update causes entangled enhancement.

## Slide 8: Experimental Setup  (about 55 s)

Next, the data set. We use 350 driving scenes from nuScenes, a public driving data set. Each scene is about 20 seconds of front-camera video. *(point to the four tiles)* We split the scenes by city. Boston 1 trains C-one. Boston 2 trains C-two and C-three. Singapore 1 is the new city, used for the updates. Singapore 2 is the test set. This split reproduces a realistic situation, in which a vehicle trained in Boston starts driving in Singapore.

*(right panel)* A random seed is the starting number for the random choices in training. We train two initial versions of C-one with two seeds. We update each version in 15 ways, which gives 30 updates. The 30 updates are not repetitions. They differ in the amount of new data, the number of epochs, the random seed and the initial C-one. All experiments run on Kaggle, and training uses one Tesla T4 GPU.

## Slide 9: Experiment for RQ1: Evaluation Procedure  (about 55 s)

Now the experiment for the first question. *(left side)* For each update, we run the whole pipeline twice on the same test scenes, once with C-one and once with C-one prime. C-two and C-three stay fixed. They are trained on the output of C-one, as in a real pipeline. We call this pipe training. Then we compare the two runs.

A small difference may come from chance. *(right side)* We therefore use three levels of evidence. Level one is any increase. Level two is an increase larger than the seed variation. Our two initial versions of C-one are almost equally accurate. Still, swapping one for the other changes the prediction error by up to 11 centimetres. We call this change the seed variation. It comes from randomness alone. Level three is statistical significance. We resample the test scenes 2,000 times. The increase must stay above zero in the middle 95 percent of the resamples.

## Slide 10: Experiment for RQ2: Two Training Modes  (about 40 s)

The second question is a separate experiment. We add a second way to train C-two and C-three. *(top lane)* In iso training, they learn from the ground truth. They are isolated from C-one and never see its errors. *(second lane)* In pipe training, they learn from what C-one really outputs, errors included.

*(point to the test box)* After each update, both versions receive exactly the same input from C-one prime. Only the training is different. We also retrain C-two and C-three on the output of C-one prime. *(bottom box)* If only the pipe version becomes worse, the cause is its adaptation to the errors of C-one.

## Slide 11: Results for RQ1: Prediction Model C2  (about 50 s)

We now turn to the results, starting with the first question. *(top right box)* Every one of the 30 updates made C-one more accurate. This plot shows delta-two, the increase of the prediction error, after each update.

*(point to the dots)* The prediction error increases in 23 of the 30 updates. Eleven dots lie above the dashed line, the seed variation. Randomness alone cannot explain these eleven. Three are circled, which means statistically significant. *(point to the left half)* All eleven clear increases come from the initial C-one with seed 0.

*(result box)* Entangled enhancement therefore occurs in the prediction model.

## Slide 12: Results for RQ1: Planning Model C3  (about 40 s)

Does the problem reach the end of the pipeline, the planned path? *(point to the plot)* The planning error increases in 18 updates, but only a little: at most 7 centimetres, against a planning error of about 4 metres. *(right boxes)* C-three relies mostly on the recent path of the vehicle. Removing all objects moves its plan by only about one metre.

*(answer box)* The answer to the first question is yes. A better C-one increases the prediction error in 23 of 30 updates, 11 of them beyond the seed variation. The planning error increases only slightly.

## Slide 13: Results for RQ2: Pipe and Iso Training  (about 50 s)

Next, the second question: why does the prediction error increase? *(point to the navy dots)* The navy dots are pipe training, as before. *(point to the gold diamonds)* The gold diamonds are iso training. Most of them lie at or below zero. In every update, the navy dot is higher than the gold diamond, although both versions receive the same input. *(table)* After iso training, no increase is larger than the seed variation.

*(result box)* The increase therefore comes from the adaptation of C-two to the errors of C-one. *(top right box)* For the planning model, the pattern is reversed. The small increase appears more often after iso training. Adaptation cannot explain it.

## Slide 14: Results for RQ2: Retraining on the Output of C1′  (about 40 s)

If adaptation is the cause, letting C-two and C-three learn the new errors should remove the increase. We retrain them on the output of C-one prime.

*(point to the bars)* The light bars show the errors before the update. The red bars show the errors after the update, with C-two and C-three kept fixed. The navy bars show the errors after we retrain them. In every group, the navy bar is the lowest.

*(answer box)* The answer to the second question is yes, for the prediction model. The increase appears only after pipe training, and retraining on the new output removes it.

## Slide 15: Conclusion  (about 30 s)

To summarize. In this study, we investigated entangled enhancement in a real autonomous driving pipeline. We updated C-one 30 times. *(RQ1 row)* First, a better C-one makes the prediction model worse in 23 of 30 updates, and clearly in 11. The planning error changes very little. *(RQ2 row)* Second, the cause is the adaptation of the prediction model to the errors of C-one. Retraining on the new output removes the increase.

## Slide 16: Future Work  (about 22 s)

Our next step is detection. *(top box)* We are building a method that detects entangled enhancement before C-one prime is used, and before ground truth for the new data exists. *(cards)* We also want more test scenes and more initial versions of C-one. The frequency we measured can also serve the retraining models of Wang and Machida.

## Slide 17: References  (about 5 s)

These are the works I referred to in this talk.

## Slide 18: Thank You  (about 5 s)

Thank you for your attention. I welcome your questions and comments.

---

# Appendix (only if asked)

## Slide 19: Appendix A: Updates of Initial C1, Seed 0  (about 10 s)

This table lists the 15 updates of the initial C-one trained with seed 0, with the same numbers as the result plots.

## Slide 20: Appendix B: Updates of Initial C1, Seed 1  (about 10 s)

This table lists the 15 updates of the initial C-one trained with seed 1, with the same numbers as the result plots.

## Slide 21: Appendix C: Details of the ML Models  (about 8 s)

This slide gives the details of the three models and of the training of C-two and C-three.

---

# Expected Questions

Short answers for the discussion. Every number comes from the results shown in the slides or the appendix.

**Q1. Are the 30 updates repetitions of the same procedure?**

No. The 30 updates use 15 settings on each of the two initial C1. The settings differ in the amount of new data (100, 50 or 25 percent of Singapore 1), the number of epochs (5, 10 or 20) and the random seed (1, 2 or 3).

**Q2. What explains the increase of the planning error after iso training?**

Adaptation cannot explain it. An iso-trained C3 learns from the true objects and their true future paths. At test time, the same C3 receives the detected objects and the paths that C2 predicts. A change in the objects that C1 prime finds can therefore move the plan of this C3 more. This study does not test this explanation directly, and the increase is at most 0.127 metres.

**Q3. Why do all clear increases come from the initial C1 with seed 0?**

The C2 trained on the output of this C1 is sensitive to a change in the errors of C1. When we replace this C1 by the other initial C1, which is almost equally accurate (0.009 apart in mAP50), the error of this C2 increases by 0.112 metres. The C2 trained on the output of the seed-1 C1 improves by 0.098 metres under the same replacement. The adaptation of C2 therefore differs between the two initial C1.

**Q4. Does the ego path let C3 ignore the objects?**

An ego-path input can let a planning model ignore the objects on nuScenes, which Li et al. and Zhai et al. report. With the ego path, removing the objects still moves the plan of C3 by 0.73 to 1.65 metres, which is above our threshold of 0.5 metres. C3 still uses the objects.

**Q5. Why is the accuracy of C1 low, with an mAP50 of about 0.11?**

C1 is trained on only 100 Boston scenes with the front camera, and the test scenes come from another city. This study needs the change of accuracy, not a high accuracy. Every one of the 30 updates improves mAP50 by 0.048 to 0.118.

**Q6. Why only 350 scenes and 50 test scenes?**

We reduced the data to 350 scenes to run 30 updates with two training modes on one GPU. With 50 test scenes, only 3 of the 30 increases are statistically significant. More test scenes are part of the future work.

**Q7. Does the accuracy gain delta one predict the increase?**

No. The rank correlation between delta one and delta two is plus 0.04. A larger accuracy gain does not come with a larger increase of the prediction error.
