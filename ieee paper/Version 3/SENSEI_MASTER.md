# Sensei review master file

Single reference for all feedback from Prof. Fumio Machida on this manuscript.
Supersedes `SENSEI_71_COMMENTS_AUDIT.md`, `SENSEI_EMAIL_AUDIT.md`,
`SENSEI_WRITING_MASTER.md` and `SENSEI_COMMENTS_0818.md`, which describe earlier
versions of the paper.

Current file: `IEEE_Conference_Paper_E1_v4.tex`, 8 pages, builds clean.

---

## Review history

| Round | Source | Scope | Status |
|---|---|---|---|
| 1 | `AIoT2026-ifty-v1-comment.pdf`, 71 PDF comments | Whole paper (the CARA version) | 71/71 neutralized |
| 1b | Email, 13 requirements | Structure and story | 13/13 satisfied |
| 2 | High-level email, no PDF | Presentation quality | Superseded by round 3 |
| 3 | `comment.pdf`, 30 PDF comments + email | Pages 1 to 4 only | **28/30 done, 2 in the abstract** |

Round 3 stopped at Section IV:

> "There are too many structural and writing issues, so I cannot read beyond
> Section IV. I would like to ask you again to review your writing yourself."

He also stated that the comments are samples, not a complete list:

> "I cannot point out everything, so my comments/suggestions are just examples
> and hints."

Every comment is therefore treated as one instance of a rule to be applied
everywhere, including Sections V to VII which he never reached.

---

## PART 1 — His four principles

### P1. Readers read top to bottom

> "Mind readers. They read from top to bottom. You need to imagine how the reader
> builds their understanding in a sequence. The specific term/concept should be
> clearly explained before they use."

A reader gets one pass and cannot look ahead. Check: for each technical term, is
its first *use* after its first *definition*?

### P2. Avoid custom terms

> "Avoid using custom terms. Carefully choose the words."

Every invented term is something the reader must memorize before they can follow
the results. Each must earn its place or be replaced with plain description.

### P3. Structure before prose

> "Do not rush writing. Create the structure first, considering reading
> trajectory. Write the list of topic sentences that should be presented in that
> section. Write a paragraph corresponding to the topic sentence."

Line-editing prose that has a structural problem only polishes the problem.

### P4. Re-read as a fresh reader

> "Review your writing as a fresh reader. Need to reset your prior knowledge when
> you review. Try to find a logical gap and implicit assumptions in the original
> writing."

### From round 2, still in force

> "An academic paper does not need fashionable, cool, or executive summary
> sentences. It is not a business document nor a blog post for engineers."

> "Imagine the situation where you need to educate junior students to follow your
> study. Simply tell what you did, observed, and proposed."

---

## PART 2 — The rules

### R1. No announcer sentences
A sentence that tells the reader something is coming instead of saying it.
Marked: `This is where a problem can appear.`
Watch: "This raises the question of", "Here we note that", "It is important to
observe that", "X is harder than Y", "Several patterns emerge", "A natural
question is".

### R2. No custom terms without justification
Marked: `cascade measurements`, `audit set`, `matrix`.
One concept, one name, for the whole manuscript. If a term is unnecessary, delete
it rather than define it.

### R3. One paragraph, one topic
Marked: the single paragraph covering $C_1$, $C_2$ and $C_3$; the Metrics
section.
If a paragraph covers N things it needs N paragraphs, with parallel openings.
**But** a paragraph under about 45 words is too short and should be merged with
its neighbour.

### R4. Say why an evaluation is needed, do not label it
Marked: the `Objective.` keyword; the six-experiment enumeration.
> "It does not need such a keyword. Rather, you need clear statement why this
> evaluation needs to do. Learn from other papers."

> "Consider a logical story why do we evaluate these things so that readers can
> also agree with these questions. This looks just an enumeration of what you
> did."

### R5. Simple, common wording
Marked: `establishes evidence`, `Six experiments serve this goal`.
Never make the reader count contributions or sections. Prefer ordinary verbs.

### R6. One sentence, one message
Marked: the shared-models sentence; an unrelated `and` join.
Do not join unrelated facts with "and". Do not stack items in a subject before
the verb. Do not bury the main claim in a trailing clause.

### R7. No logical gaps
Marked: the exact-equality sentence; the safety-term chain; the pipeline-only
claim.
If a "therefore" skips a step, write the step.

### R8. Introduce a section before its content
Marked: the start of Section III.
Objective, then overview, then content.

### R9. Equation placement
Marked: rephrase, then insert the expression; move the equation up; delete the
sentence that only announces it.
Punctuation follows the sentence: comma if it continues, full stop if it ends.

### R10. A list contains only what it claims to
Marked: setup steps inside a "Retraining Procedure" list.

### R11. Formatting
Marked: `$20$~s` should be `$20$\,s`.
No em dashes. No prose colons or semicolons. Headings take a colon, never a full
stop. Always `$C_1$`, never bare `C1`.

### R12. Title names what the paper is
Strikeouts on `Tackling` and `the Model Pipeline for`, carets inserting
`An Empirical Analysis of`. No verb-first slogans.

---

## PART 3 — Round 3 comment status

### Page 1
| Comment | Target | Status |
|---|---|---|
| Title strikeouts and carets | title | done |
| "Avoid this kind of phrase" | `This is where a problem can appear.` | done |
| "? Need to revise" | `This kind of maintenance is possible only in a pipeline...` | done |

### Page 3
| Comment | Target | Status |
|---|---|---|
| "Introduce the context before jump into the content" | start of Section III | done |
| "Do not need space" | `~` before units | done, 18 fixed |
| "do not use unclear term" | `cascade measurements` | done |
| "Not clear from the previous sentences" | `audit set` | done |
| strikeout | `that $C_2$ consumes` | done |
| "Change the paragraph here" (x2) | the three-model paragraph | done |
| "Rephrase this. Then, insert Expression 9" | the two-terms sentence | done |
| "Move above." | Eq. 9 | done |
| strikeout `its`, "?correct?" on `scorer` | PDM-Closed sentence | done |
| "Need to revise" | the safety-term chain | done |
| "Is this as the step of Retraining Procedure?" | step 1 | done |
| "Overall, unclear." | steps 2 to 5 | done |
| "Simple things are not presented in an organized way" | Metrics | done |

### Page 4
| Comment | Target | Status |
|---|---|---|
| "What is matrix." | `The matrix combines...` | done |
| "Not a good sentence" | the shared-models sentence | done |
| "Why do we need to connect... with and" | the seed sentence | done |
| "The meaning is unclear." | exact equality vs statistical closeness | done |
| "Use more simple sentence" | `establishes evidence...second contribution` | done |
| "This kind of phrasing is also uncommon" | `Six experiments serve this goal` | done |
| "Consider a logical story" | the six-experiment enumeration | done |
| "It does not need such a keyword" | `Objective.` | done, all 6 removed |

---

## PART 4 — Work completed beyond his comments

Applying each rule to the sections he never reached.

**Custom terms removed**

| Term | Action |
|---|---|
| cascade gap, $\Gamma_k$ | Definition and equation deleted; Section II-C retitled "Two-Mode Measurement and the Isolation Property" |
| effect vector | deleted |
| cascade signature | deleted |
| correction-cascade pattern | replaced with an attribution to Sculley et al. |
| kinematic prior | "its own speed and lane rules" |
| ego-forecasting shortcut | "a known weakness of open-loop L2 on nuScenes" |
| regime, masked sensitivity, zero-delta, campaign-dependent | plain wording |

**Kept**: entangled enhancement (from wang2024compsac), isolated and pipeline
mode, isolation property, coupling factor, plan shift.

**Structural**

- Section IV overview rewritten as a chain of doubts rather than a list.
- Section IV-B retitled "Ruling Out a Weaker Detector".
- Training Diagnostics given a purpose and added to the overview.
- Section V-B mitigation paragraph deleted as a duplicate of Section VI-C.
- The drift-screen experiment given the setup sentence it never had.
- Conclusion split into four paragraphs and the class-balanced campaign added.
- Seven over-short paragraphs merged back; nothing under 45 words remains.

**Consistency**

- Collision rate written as `0.020` everywhere, matching Table I.
- Equation punctuation audited; Eq. 5 was missing its full stop.
- Casual register swept from Sections II to VII.

---

## PART 5 — Open items

1. **Abstract not yet revised.** Still says "real entangled enhancement" twice,
   attributes the cause to "distribution shift" rather than to downstream models
   tuned to the previous upstream output, names entangled enhancement before
   defining it, and reports 15 updates when the paper now reports 15 plus 12.
2. **Section VI Related Work** has not had a full pass against these rules.
3. **The coupling factor $\rho$** is claimed as a contribution but no value is
   reported. Section II-D now says it can be computed from the $\delta_1$ and
   $\Delta_3$ columns of Table II. A reviewer may still ask why it is a
   contribution if never used. Adding a column would mean editing a table, which
   the user has ruled out.
4. **The $+0.008$\,m case** is described as near the resolution of the
   measurement. An incumbent-versus-incumbent run would establish the noise
   floor. Deferred by the user.

---

## PART 6 — Per-batch checklist

- [ ] Every changed sentence carries one message (R6).
- [ ] Every pronoun has an unambiguous antecedent.
- [ ] No new term without a definition at first use (R2).
- [ ] Terms swept across the whole .tex including figures and captions.
- [ ] No paragraph under about 45 words (R3).
- [ ] Clean grep for: em dash, prose `;`, prose `:`, bare `C1/C2/C3`, `~unit`.
- [ ] No data, table, figure or data explanation edited or deleted.
- [ ] Build clean, page count 8.
- [ ] Temp files cleared after every compile.
