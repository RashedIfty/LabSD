# Sensei's writing rules

Derived from the 30 PDF comments on `comment.pdf` (review of 2026-08-18) and the
accompanying email. His review stopped at Section IV: "There are too many
structural and writing issues, so I cannot read beyond Section IV."

He states explicitly that the comments are samples, not a complete list:

> "I cannot point out everything, so my comments/suggestions are just examples
> and hints."

So each comment below is one instance of a rule that must be applied everywhere
in the paper, including the sections he never reached (V, VI, VII).

---

## The four principles from his email

### 1. Mind readers, they read top to bottom

> "You need to imagine how the reader builds their understanding in a sequence.
> The specific term/concept should be clearly explained before they use."

A reader gets one pass and cannot look ahead. Every term must be defined before
its first use, and every claim must be usable at the point it appears.

**Check:** for each technical term, is its first *use* after its first
*definition*?

### 2. Avoid custom terms

> "Avoid using custom terms. Carefully choose the words."

Every invented term is something the reader must memorize before they can follow
the results. Each one must earn its place or be replaced with plain description.

### 3. Structure first, then prose

> "Do not rush writing. Create the structure first, considering reading
> trajectory. Write the list of topic sentences that should be presented in that
> section. Write a paragraph corresponding to the topic sentence."

Write the sequence of topic sentences for a section, confirm that sequence reads
as an argument on its own, and only then write the paragraphs. Line-editing prose
that has a structural problem only polishes the problem.

### 4. Re-read as a fresh reader

> "Review your writing as a fresh reader. Need to reset your prior knowledge when
> you review. Try to find a logical gap and implicit assumptions in the original
> writing."

Hunt for the places where you know something the reader does not, and never said
it.

---

## The rules, with evidence

### R1. No announcer sentences

A sentence that tells the reader something is coming, instead of saying it.

| Comment | Text he marked |
|---|---|
| "Avoid this kind of phrase" | `This is where a problem can appear.` |

**Same class, found and removed:**
- `The reason is that those models were built...` → `Those models were built...`
- `Before any of these changes is interpreted, C2 and C3 must be shown to be untouched.`
- `The check validates the measurement setup rather than reporting a result.`
- `The reference update is a single point in a large space of possible updates.`
- `...and three patterns emerge.`
- `A natural question is whether...` → `The effect could still be...`
- `Each model is scored with the metric standard for its task.`
- `One further quantity is recorded for the campaign.`
- `The experiment is prepared once.`
- `Every retraining event then follows the same three steps.`

**Watch for:** "This raises the question of", "Here we note that", "It is
important to observe that", "This has an important implication", "Detecting the
condition is harder than defining it."

### R2. No custom terms without justification

| Comment | Text he marked |
|---|---|
| "do not use unclear term" | `cascade measurements` |
| "Not clear from the previous sentences" | `audit set` |
| "What is matrix. You have never told about matrix before this." | `matrix` |

**Fixed:** `cascade measurements` → `measurements`; `audit set` → `subset`;
`The matrix combines...` → `These three factors are combined into...`

**Still in the paper, each needing a decision:**

| Term | Status |
|---|---|
| entangled enhancement | Keep. Inherited from wang2024compsac. |
| isolated / pipeline mode | Keep. Genuinely needed, well defined. |
| isolation property | Keep. It is a verified control. |
| cascade gap (Gamma_k) | **Cut candidate.** Defined in II-C, next used 300 lines later. |
| cascade coupling factor (rho) | **Cut candidate, strongest.** Claimed as a contribution, never reported in Section IV. |
| plan shift | Keep. Appears in a table. |
| effect vector | **Cut.** Used once. |
| cascade signature | **Cut.** Fig. 1 caption, undefined, used once. |
| correction-cascade pattern | **Cut or attribute.** Sculley's term, used as if defined. |

### R3. One paragraph, one topic

| Comment | Target |
|---|---|
| "Change the paragraph here" (x2) | The single paragraph covering C1, C2 and C3 in III-A |
| "Simple things are not presented in an organized way... Only one sentence for C1. Everything is in one paragraph, despite three different metrics are introduced for three components." | Section III-C Metrics |

**Fixed:** III-A split into three paragraphs, one per model, in pipeline order.
III-C split into five: C1 (one sentence), C2, C3, plan shift, protocol caveat.

**Rule:** if a paragraph covers N things, it needs N paragraphs. Keep the
openings parallel so a reader can find any one at a glance.

### R4. Say why an evaluation is needed, do not label it

| Comment | Text he marked |
|---|---|
| "It does not need such a keyword. Rather, you need clear statement why this evaluation needs to do. Learn from other papers." | `Objective.` |
| "Consider a logical story why do we evaluate these things so that readers can also agree with these questions. This looks just an enumeration of what you did." | The six-experiment list in Section IV |

**Fixed:** all six `\emph{Objective.}` labels removed; each subsection now opens
with the reason the check is needed. The Section IV overview was rewritten as a
chain in which each experiment answers a doubt the previous one raises.

**Rule:** the reader should agree the experiment was necessary before reading its
result. A list of what you did does not achieve that.

### R5. Simple, common wording

| Comment | Text he marked |
|---|---|
| "Use more simple sentence. Why do you state like this way. 'establishes evidence' is uncommon wording. 'which is the second contribution' is unnecessary at all." | `The evaluation establishes evidence of entangled enhancement in the AV pipeline, which is the second contribution.` |
| "This kind of phrasing is also uncommon. Write more straightforward simple sentences." | `Six experiments serve this goal,` |

**Fixed:** → `This section reports whether entangled enhancement occurs in the AV
pipeline.` and `We run six experiments.`

**Rule:** never make the reader count contributions or sections. State the thing.
Prefer ordinary verbs (is, has, shows) over constructed ones (establishes
evidence, serves this goal).

### R6. One sentence, one idea

| Comment | Text he marked |
|---|---|
| "Not a good sentence" | `The original C1 (Boston-trained) and the unchanged C2 and C3 are shared across all 15 updates, and only the Singapore fine-tuned C1' changes between them.` |
| "Why do we need to connect the previous sentence with 'and'" | `...uses a single fixed random seed, and the evaluation of C2 and C3 is deterministic because neither model has trainable parameters.` |

**Fixed:** both split. The important claim now comes first: `Only the Singapore
fine-tuned C1' differs between the updates.`

**Rule:** do not join unrelated facts with "and". Do not stack three items in a
subject before reaching the verb. Do not bury the main claim in a trailing
clause.

### R7. No logical gaps

| Comment | Text he marked |
|---|---|
| "The meaning is unclear." | `The isolation property (6) can therefore be checked for exact equality rather than for statistical closeness.` |
| "Need to revise" | `The safety term is evaluated on z2... hence the safety cost, hence which candidate wins.` |
| "? Need to revise" | `This kind of maintenance is possible only in a pipeline...` |

**Fixed:** the isolation sentence now gives the middle step (determinism →
repeated measurement reproduces the same numbers → property verified by exact
equality). The safety-term chain was rewritten as an explicit causal path.

**Rule:** if a "therefore" skips a step, the reader has to supply it. Write the
step.

### R8. Introduce a section before its content

| Comment | Target |
|---|---|
| "Introduce the context before jump into the content. (objective, overview, etc)" | Start of Section III |

**Fixed:** Section III now opens with its objective and a roadmap of the four
subsections.

**Rule:** state the objective, then the overview, then the content. Applies to
every section.

### R9. Equation placement

| Comment | Text he marked |
|---|---|
| "Rephrase this. Then, insert Expression 9" | `it gives each candidate a cost made of two terms.` |
| "Move above." | Equation 9 |
| StrikeOut | `The executed plan is the candidate of minimum total cost,` |

**Fixed:** equation moved up to follow the sentence that motivates it; the
separate introducing sentence deleted; the term explanations now follow the
equation.

**Rule:** the equation follows the sentence that motivates it. Do not spend a
whole sentence announcing an equation.

### R10. A list must contain only what it claims to

| Comment | Text he marked |
|---|---|
| "Is this as the step of Retraining Procedure? It does not look 'retraining'" | Step 1, `Train C1 on the source domain... freeze C2 and C3` |
| "Overall, unclear." | Steps 2 through 5 |

**Fixed:** one-time setup moved out of the numbered list into prose; the list cut
from five steps to three, containing only retraining.

### R11. Formatting

| Comment | Target |
|---|---|
| "Do not need space" | `$20$~s`, `$2$~Hz` |

**Fixed:** all 18 instances of `~unit` converted to `\,unit`.

### R12. Title

StrikeOut on `Tackling`, `the Model Pipeline for`, and the trailing `s`, with
carets inserting `An Empirical Analysis of` and `Model Pipeline`.

**Result:** `An Empirical Analysis of Entangled Enhancement in Autonomous Vehicle
Model Pipeline`

**Rule:** the title names what the paper is, not what it does to the problem. No
verb-first slogans.

---

## Standing constraints from earlier rounds

- No em dashes.
- Never end a heading with a full stop; use a colon.
- Do not reference the previous version of the paper or the small-scale
  experiment.

---

## Open items he has not seen

He stopped at Section IV, so these are unreviewed and likely to draw the same
comments:

1. **rho is claimed as a contribution but never reported.** No rho value appears
   anywhere in Section IV. Add a rho column to Table II or remove it from the
   contributions.
2. **The 12-update class-balanced campaign is never introduced.** The abstract,
   introduction and conclusion all say 15. A reader reaching IV-F cannot place
   those 12 runs.
3. **The drift-screen experiment in Section V has no setup.** Rank correlation
   0.56 and AUROC 0.48 appear with no prior mention in Section III and no place
   in the Section IV overview.
4. **Section IV has seven subsections but the overview promises six.** Training
   Diagnostics is unlisted.
5. **Section IV-B's heading is a claim, not a topic:** "The Detector Improves,
   Yet the System Degrades". Its job is ruling out an alternative explanation.
6. **Section V paragraph 3 duplicates Section VI-C**, same four citations.
7. **The mechanism paragraph in Section V-A assumes** the human driver did not
   react to the agents the planner started reacting to. Unstated.
8. **Units:** text says "2%", Table I says "0.020", the delta column says "n/a".
9. **The +0.008 m case** is likely below measurement resolution but is counted as
   entangled enhancement without qualification.
