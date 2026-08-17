# Sensei (Machida) — email requirements audit

The email broken into 13 discrete requirements, each verified against the current
`IEEE_Conference_Paper_E1_v4.tex`. **13/13 satisfied.**

| # | Email requirement | How it is satisfied |
|---|-------------------|---------------------|
| E1 | Contributions summarized in three items (his exact wording) | Intro lists exactly: (1) a formal definition of entangled enhancement in the perception-prediction-planning pipeline for autonomous vehicles, (2) empirical evidence of entangled enhancement in the AV pipeline system, (3) a proposal of CARA, a mitigation technique |
| E2 | Abstract reformulated to present the contributions as a clear story | Abstract states "three contributions" and walks First / Second / Third |
| E2b | Introduction reformulated | F1-F7 findings list removed; intro now motivates the gap and ends on the 3 contributions |
| E3 | Subsequent sections follow the main story | II = formal definition (contribution 1), III = CARA (contribution 3), V = evidence (contribution 2) + CARA evaluation (contribution 3) |
| E4 | "Section II provides the formal definition" | Section II keeps the title "System Model and Methodology" (author's choice, matching sensei's outline item "2. System model"); its experimental content moved to Section IV, and it now contains the formal definition of entangled enhancement |
| E5 | "dual-mode evaluation" is NOT a contribution; present it as a formulation of EE | Term "dual-mode" removed everywhere (0 occurrences); the two-mode concept now opens with "To make entangled enhancement measurable, we separate a model's own error from the error it inherits" inside the formal definition |
| E6 | Section III needs more careful description of the method | Rewritten with explicit inputs/outputs and a three-step procedure tied to Fig. 2 and Alg. 1 |
| E7 | Section III must not rely on later sections | 0 references to Section V results, tables, or findings inside Section III |
| E8 | Section V revised following the structural changes | Restructured 10 -> 8 subsections; all (F#) tags removed; redundant restatements merged |
| E9 | Findings not in arbitrary order; needs an overview | New overview states the two evaluation goals and names all six experiments up front |
| E10 | Explain the objectives of the individual experiments | Every substantive subsection opens with an italic *Objective.* sentence (7 total) |
| E11 | Everything tightly/logically connected to the main story | The overview maps goal 1 to "the second contribution" and goal 2 to "the third contribution" explicitly |
| E12 | Many sentences colloquial/informal; polish | 0 "so" conjunctions, 0 prose semicolons/colons, 0 contractions, 0 hedges ("in fact"), 0 sentence-initial "And", vague "This is" subjects rewritten, 0 "modular", 0 "incumbent", 0 "strict" |

## Final section structure (maps to the story)
```
I    Introduction                      -> the 3 contributions
II   System Model and Methodology      -> contribution 1 (formal definition of EE)
III  Cascade-Aware Retraining Assessment -> contribution 3 (method)
IV   Experimental Setup
V    Evaluation Results                -> contribution 2 (evidence) + contribution 3 (evaluation)
VI   Discussion
VII  Related Work
VIII Conclusion
```

### Deviations from sensei's later outline (author's explicit decisions)
Sensei's outline was: 1 Introduction, 2 System model, 3 CARA, 4 Experiment setup,
5 Evaluation results, 6 Related work, 7 Conclusion (no Discussion), with the
introduction ordered EE issue -> AV pipeline -> gap -> this paper.
- **Discussion section kept** (author's decision) rather than folded into Sec V and
  Related Work. Section count is 8 instead of 7.
- **Introduction order kept** (author's decision) as AV pipeline -> EE issue + gap ->
  this paper, rather than leading with the EE issue.
- **Section II title kept** as "System Model and Methodology" (author's decision).
These are the only three places where the manuscript departs from sensei's outline.

## Extra items caught during the email pass (not in the 71 comments)
- "strict" survived in three places after the definition fix (Section V, admission
  rule, Conclusion). All removed or reworded.
- One sentence-initial "And", one "in fact" hedge, and four vague "This is" subjects
  rewritten for the formality sensei asked for.

## Still open (needs the author)
1. Three citation placeholders: `TODO_neurips2021_selfdefeating`,
   `TODO_sentence_classification`, `TODO_context13`.
2. No limitations / threats-to-validity section. The LiDAR versus camera caveat that
   sensei struck from Section IV would naturally live there.
