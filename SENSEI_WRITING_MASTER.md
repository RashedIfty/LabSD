# Sensei's writing rules: master file

Every writing rule that Prof. Fumio Machida has given, in one place. Use this file for every report, paper and email to sensei. It applies to all future writing, not only the paper it came from.

**Sources:**
- Round 1: 71 PDF comments on the paper, plus an email listing 13 requirements
- Round 2: a high-level email
- Round 3: 30 PDF comments (`comment.pdf`, 2026-08-18) and an email
- Meeting 6 report: two emails (2026-09-24)
- Rules the author set during the same work

The older per-round files are still in `ieee paper/Version 3/`, as `SENSEI_*.md`.

Sensei says his comments are samples, not a complete list: "I cannot point out everything, so my comments/suggestions are just examples and hints." Treat every comment as a rule and apply it to the whole document.

---

## Part 1: Principles

**P1. Readers read top to bottom.**
> "Mind readers. They read from top to bottom. You need to imagine how the reader builds their understanding in a sequence. The specific term/concept should be clearly explained before they use."

Define every term before its first use. Do not rely on later sections. Do not make the reader look ahead.

**P2. Avoid custom terms.**
> "Avoid using custom terms. Carefully choose the words."

If a term isn't needed, delete it instead of defining it. Use one name for one concept across the whole document, and never switch to a synonym for variety.

**P3. Plan the structure before writing the prose.**
> "Do not rush writing. Create the structure first, considering reading trajectory. Write the list of topic sentences that should be presented in that section. Write a paragraph corresponding to the topic sentence."

**P4. Re-read as a fresh reader.**
> "Review your writing as a fresh reader. Need to reset your prior knowledge when you review. Try to find a logical gap and implicit assumptions in the original writing."

**P5. Write academically and plainly.**
> "An academic paper does not need fashionable, cool, or executive summary sentences. It is not a business document nor a blog post for engineers."
> "Imagine the situation where you need to educate junior students to follow your study. Simply tell what you did, observed, and proposed."
> "We need very careful writing, sentence by sentence."

**P6. A report is practice for writing a paper.**
> "The current report is in the form of an oral presentation instead of an academic report... please follow the common paper format... The academic report should be readable for other relevant researchers."

---

## Part 2: Structure

**S1. Use the common paper format.** Abstract, Introduction, Background or Related Work, the core sections (for example Hypotheses, Method or Experimental Design, Results), Conclusion, References.

**S2. Use standard section titles.** Never use presentation-style titles such as "Where the Study Stands", "Checks That Stay in Place", "What Would Make the Hypothesis Fail", "Why This Hypothesis" or "What Follows From It". A title must state what the section is for. A title must not state a finding: "Ruling Out a Weaker Detector", not "The Detector Improves, Yet the System Degrades".

**S3. Introduce every section before its content.** Start with the objective, then an overview, then the content.
> "Introduce the context before jump into the content. (objective, overview, etc)"

**S4. One paragraph, one topic.** If a paragraph covers N things, it needs N paragraphs, with openings that follow the same pattern. For example, write one paragraph per component, never C1, C2 and C3 together. A paragraph under about 45 words is too short, and should be merged with its neighbour.

**S5. Explain why an evaluation is needed. Don't just label or list it.**
> "It does not need such a keyword ["Objective."]. Rather, you need clear statement why this evaluation needs to do."
> "Consider a logical story why do we evaluate these things so that readers can also agree with these questions. This looks just an enumeration of what you did."

**S6. A list contains only what its title claims.** Setup steps don't belong in a list called "Retraining Procedure".

**S7. Keep supplemental material out of the main story.**

**S8. Place equations right after the sentence that motivates them.** Don't spend a whole sentence announcing an equation. Put the explanation of each term after the equation. Punctuate the equation as part of the sentence: a comma if the sentence continues, a full stop if it ends.

**S9. The title names what the work is.** "An Empirical Analysis of..." is right. Never start with a verb slogan such as "Tackling...".

**S10. Present the contributions as a clear story.** List them in the Introduction. Never make the reader count them ("which is the second contribution").

---

## Part 3: Sentences

**W1. One sentence, one message.** Do not join unrelated facts with "and". Do not stack several items before the verb. Do not hide the main claim in a trailing ", which..." clause. A ", which is..." clause that only defines a term is fine.

**W2. No announcing sentences.** A sentence must say the thing, not announce that it is coming.
- Sensei marked: "This is where a problem can appear."
- Watch for: "This raises the question of", "Here we note that", "It is important to", "A natural question is", "Several patterns emerge", "The question that matters is", "This report prepares".

**W3. Use simple, common wording.** Prefer plain verbs such as is, has and shows.
- Sensei marked: "establishes evidence" and "Six experiments serve this goal".
- Use: "This section reports whether..." and "We run six experiments."

**W4. No logical gaps.** If "therefore" skips a step, write the missing step.

**W5. No vague pronouns.** Avoid "it", "its", "they", "this", "these", "the two" and "before it" unless the noun they refer to is obvious. Write the noun instead, for example "the preceding module".

**W6. Don't use "so" as a joining word.** Rewrite the sentence.

**W7. Sentences must stand alone.** Don't write short, clipped sentences that assume the reader already knows the point.

**W8. Keep the tone formal.** No contractions. No sentence-initial "And". No hedges such as "in fact". No decorative qualifiers such as "fully controlled".

---

## Part 4: Content to leave out

**C1. No meeting or lab context.** Do not write "At the last meeting, Prof. Machida set three tasks", "Task 2", "From Meeting 1..." or anything else about earlier meetings. Say that at the meeting.

**C2. No high-context remarks.** Remove any sentence about the review comments or the rebuttal of the base papers. Sensei: "no one can understand".

**C3. Do not mention previous versions of the paper**, or the small-scale early experiment.

**C4. Only state claims the evidence supports.** A hypothesis says "can", not "always". Sensei: "It is not always the case that planning error increases... The hypothesis is doubtful."

---

## Part 5: Fixed wording

| Never write | Write |
|---|---|
| our car | ego vehicle |
| our paper | this study |
| learned models | ML models or ML components |
| error grows | error increases |
| planner, detector, predictor | planning module, perception module, prediction module |
| camera detector (for YOLO) | an object detector applied to the front-camera image |
| "C1" in plain text | `$C_1$`, `$C_2$`, `$C_3$` |
| machine learning system | MLS (spell it out once, then use MLS only) |
| modular | pipeline |
| down stream | downstream |
| incumbent | original |
| "the original $C_1$" / "the updated model" (figure boxes) | "original" / "updated" |
| fine-tuned model (unnamed) | "the Singapore fine-tuned $C_1'$" |
| $R$ (for a retrained model) | $C_i$ and $C_i'$ |
| tolerance, $\tau_n$ | threshold |
| terminal metric, terminal effect, terminal vector, terminal readings | planning metric, or name the metric |
| strict (in the definition) | (delete) |
| bit-exactly | (delete), or "exactly the same, digit by digit" |
| apparatus | measurement setup |
| dual-mode, dual-mode evaluation, dual-mode profile | (delete); "isolated mode" and "pipeline mode" |
| two-mode profile, five-quantity profile, profile | "the metrics of $C_2$ and $C_3$, each evaluated in isolated mode and in pipeline mode" |
| isolated metrics, pipeline metrics | "the metric of $C_k$ in isolated mode / in pipeline mode" (a metric is a metric; the mode is how it is evaluated) |
| cascade measurements | measurements |
| audit set | subset (define it: fixed scenes, with ground truth, held constant) |
| The matrix combines... | These three factors are combined into... |
| cascade gap, $\Gamma_k$ | (delete) |
| effect vector | (delete) |
| cascade signature | (delete) |
| correction-cascade pattern | attribute it: "the pattern described by Sculley et al." |
| intermediate signals, pipeline interfaces, cascade channel | "the output of $C_1$", "the output of $C_2$" |
| injected effect | the change caused by the update |
| kinematic prior | "its own speed and lane rules" (or the plain description) |
| ego-forecasting shortcut | "a known weakness of open-loop L2 on nuScenes" |
| regime, entangled-enhancement regime | plain wording, e.g. "the updates that show entangled enhancement" |
| masked sensitivity, zero-delta, campaign-dependent | plain wording |
| interface-drift score, drift score, drift screen, drift proxy, front-end | pick one name, define it once, use it everywhere |
| establishes evidence | shows |
| serves this goal | plain verb ("We run six experiments.") |
| Objective. (label) | (delete); open with why the evaluation is needed |
| fully controlled | (delete) |
| end-to-end literature | name the works or cite them |
| Tackling ... (title) | An Empirical Analysis of ... |
| the Model Pipeline for ... Vehicles (title) | ... Model Pipeline |
| F1 to F7 finding tags | (delete) |
| The reason is that X | X |
| A natural question is whether | The effect could still be... |
| This is where a problem can appear. | (delete; state the problem) |
| agent | object (for things around the ego vehicle) |
| planning agent / planner | planning module |
| learned mistakes, uneven improvement | the first cause, the second cause (described in plain words) |
| isolated / old-pipeline / new-pipeline training | "learn from the ground truth / the output of the old $C_1$ / the output of the new $C_1$" |
| validity controls | checks on the measurement |
| disagreement rate | "the fraction of objects that only one of the two versions finds" |
| maintainer | the engineer who maintains the MLS |

Terms that are kept, because they are defined and needed: entangled enhancement (from Wang and Machida), isolated mode, pipeline mode, isolation property, coupling factor, plan shift, symptom (sensei's own word, defined at first use).

Other words to avoid unless defined, or to replace with plain words: ego (on its own), arm, keyframe, trajectory, campaign, tracker, noise floor, pilot, contamination, diverge, symmetrically, "which is the point".

---

## Part 6: Formatting

**F1.** Put a thin space before units: `$20$\,s`, `$4.6$\,GB`. Never `$20$~s`. Sensei: "Do not need space."
**F2.** No colons or semicolons in running text. Write lists as sentences.
**F3.** No em dashes or en dashes used as punctuation.
**F4.** An inline heading (`\paragraph{...}` or a bold lead-in) ends with a colon. Never a full stop, never a comma inside. Put a qualifier in parentheses: "H1 (existence):".
**F5.** No italics in the body text. The author set this rule on 2026-09-24; the reference-list style may keep its own.
**F6.** Put keywords in alphabetical order.
**F7.** Clear the LaTeX temp files after every compile.

---

## Part 7: Research-content guidance (Meeting 6 emails)

- **H1:** state what can happen, not what always happens. The simple hypothesis to confirm: "the update of the perception model can adversely impact the performance of prediction and planning models."
- **The fundamental question:** how to find a hint or symptom of entangled enhancement. Consider metrics that describe the change in the output distribution of $C_1$ and $C_2$. Accuracy metrics may not be available in deployment.
- **A frequency-only hypothesis** (how often it happens) is fine for an empirical study, but it gives no clue towards a solution. Prefer questions that lead to a method.
- **Test several pipelines** with different combinations of models. Entangled enhancement is probably unavoidable in every pipeline.
- **Every module is an ML component.** No fixed-rule modules, and no fixed-rule glue steps between modules.

---

## Part 8: Checklist before sending

- [ ] The structure is planned as topic sentences first (P3).
- [ ] Every section opens with its objective (S3). Every paragraph has one topic and at least about 45 words (S4).
- [ ] Every term is defined before its first use, and uses one name throughout (P1, P2).
- [ ] No reference points to a later section, except the roadmap in the Introduction (P1).
- [ ] Every sentence carries one message (W1). Every pronoun clearly refers to a noun (W5).
- [ ] No meeting context, no remarks about reviewers, no presentation-style titles (C1, C2, S2).
- [ ] Grep comes back clean for:
  - `\bso\b`, `—`, `–`
  - prose `;` and `:`
  - `$~` before units
  - bare `C1`
  - `\emph`
  - `our car`, `our paper`, `learned model`, `grow`, `Meeting`, `Prof.`, `reviewer`
- [ ] Data, tables and figure results are unchanged unless explicitly asked.
- [ ] The document builds cleanly, and the temp files are cleared.
