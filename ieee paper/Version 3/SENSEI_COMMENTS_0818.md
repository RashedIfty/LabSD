# Sensei comments on comment.pdf (0818 round)

## p1 [Highlight]
- TEXT: This kind of maintenance is possible only in a pipeline. Each model can be retrained on its own. In an end-to-end network the whole network must be retrained at once. We therefore adopt the pipeline as the setting for this study.
- NOTE: ? Need to revise

## p1 [Highlight]
- TEXT: This is where a problem can appear.
- NOTE: Avoid this kind of phrase

## p1 [StrikeOut]
- TEXT: Tackling

## p1 [Caret]
- NOTE: An Empirical Analysis of

## p1 [StrikeOut]
- TEXT: the Model Pipeline for

## p1 [StrikeOut]
- TEXT: s

## p1 [Caret]
- NOTE: Model Pipeline

## p3 [Text]
- NOTE: Introduce the context before jump into the content. (objective, overview, etc)

## p3 [StrikeOut]
- TEXT: 2

## p3 [Text]
- NOTE: Do not need space

## p3 [Highlight]
- TEXT: cascade measurements
- NOTE: do not use unclear term

## p3 [Highlight]
- TEXT: audit set
- NOTE: Not clear from the previous sentences

## p3 [StrikeOut]
- TEXT: that C2 consumes.

## p3 [Text]
- NOTE: Change the paragraph here

## p3 [Text]
- NOTE: Change the paragraph here

## p3 [Highlight]
- TEXT: it gives each candidate a cost made of two terms.
- NOTE: Rephrase this.Then, insert Expression 9

## p3 [StrikeOut]
- TEXT: The executed plan is the candidate of minimum total cost,

## p3 [Highlight]
- TEXT: al c λ � st, ϕ � ) � π⋆ (9) = arg min π∈P Jprog( (π) + λ � ϕ � d(π, a) � . mi π∈ in ∈P a∈ ∈z2
- NOTE: Move above.

## p3 [StrikeOut]
- TEXT: its

## p3 [Highlight]
- TEXT: scorer.
- NOTE: ?correct?

## p3 [Highlight]
- TEXT: The safety term is evaluated on z2, 2 and that dependence is the channel through which a perturbed detection reaches the plan. A detection that moves, appears, or disappears changes z2, hence the safety cost, hence which candidate wins.
- NOTE: Need to revise

## p3 [StrikeOut]
- TEXT: ever

## p3 [Highlight]
- TEXT: 1) Train on the source domain (Boston scenes), and C1 freeze and permanently. C2 C3
- NOTE: Is this as the step of Retraining Procedure??It does not look "retraining"

## p3 [Highlight]
- TEXT: 2 3 Measure both modes on the target-domain validation set (Singapore). The measurement records detection C1 quality (mAP@50), the minADE of 1 in isolated and C2 pipeline mode, and the L2@{ 2 }s of in isolated {1, 2, 3}s C3 and pipeline mode. 3) Fine-tune on the target domain (Singapore scenes) to C1 in place. C′ 1 1 5) Verify the isolation property (6) and compute the changes (7) at and C3. C2 C′ 1 1 4) Repeat the identical measurement with C′ 1 . 1 obtain the updated model C′ 1
- NOTE: Overall, unclear.

## p3 [Highlight]
- TEXT: is scored with standard detection metrics (mAP@50, C1 1 mAP@50-95, precision, recall) on the target-domain validation images. For C2, the metric is the minimum average 2 displacement error (minADE). For an agent with forecast (m) nt e ˆ(m) positions ˆp( t ⋆ t over t future steps, T T � T 1 (10) minADE = min m , ⋆ t �� 2 T min m t=1 e⋆ t s, (11) L2@τ = , τ ∈{1 {1, 2, 3} e⋆ τ �� 2 C′ 1 1 moves the driving decision independent of the human path. co ��e ad �� 2 � ��p �� 1 on the same scene, which measures how far a retraining C′ 1 and the plan produced with the updated C1 with the original also report a plan shift, the distance between the plan produced reported together with the collision rate. For the campaign we ��ˆeτ − ˆeτ e⋆ τ t seconds ahead of the current frame, τ , measured e⋆ t and the human driver’s recorded trajectory ˆet ˆet ego trajectory 3 the metric is the L2 displacement error between the planned For C3, therefore collapses to that mode. min mode, and the averaged over agents. Our deterministic predictor has a single ˆ t − ��ˆp( t p⋆ t ˆ(m) ˆ t and ground-truth positions m under mode p⋆ t
- NOTE: Simple things are not presented in an organized way...Keep them simple and explain carefully.Only one sentence for C1.Everything is in one paragraph, despite three different metrics are introduced for three components.

## p4 [Highlight]
- TEXT: matrix
- NOTE: What is matrix. You have never told about matrix before this.

## p4 [Highlight]
- TEXT: C′ 1 changes between them. 3 fine-tuned C′ 1 2 updates, and only the Singapore 15 1 are shared across all ′ C3 and C2 (Boston-trained) and the unchanged C1 The original
- NOTE: Not a good sentence

## p4 [Highlight]
- TEXT: 2 3 model has trainable parameters. is deterministic because neither C3 and C2 the evaluation of
- NOTE: Why do we need to connect the previous sentence with "and"

## p4 [Highlight]
- TEXT: statistical closeness. can therefore be checked for exact equality rather than for The isolation property (6)
- NOTE: The meaning is unclear.

## p4 [Highlight]
- TEXT: The evaluation establishes evidence of entangled enhancement in the AV pipeline, which is the second contribution.
- NOTE: Use more simple sentence. Why do you state like this way."establishes evidence" is uncommon wording. "which is the second contribution" is unnecessary at all.

## p4 [Highlight]
- TEXT: Six experiments serve this goal,
- NOTE: This kind of phrasing is also uncommon. Write more straightforward simple sentences.

## p4 [Highlight]
- TEXT: The reference update (Section IV-A) asks whether entangled enhancement occurs at all in a single update. The detector analysis (Section IV-B) asks whether the planning degradation could instead be explained by a weaker detector. The isolation control (Section IV-C) asks whether the observed change can be attributed to the update
- NOTE: Consider a logical story why do we evaluate these things so that readers can also agree with these questions.This looks just an enumeration of what you did.

## p4 [Highlight]
- TEXT: rather than to the measurement setup. The cascade location analysis (Section IV-D) asks where along the pipeline the change caused by the retraining is largest. The campaign (Section IV-E) asks how often, and how strongly, entangled enhancement occurs across updates rather than one. Finally, 15 the class-balance study (Section IV-F) asks whether the effect is an artifact of class imbalance in the training data.

## p4 [Highlight]
- TEXT: Objective.
- NOTE: It does not need such a keyword.Rather, you need clear statement why this evaluation needs to do.Learn from other papers.
