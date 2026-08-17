# Sensei writing-revision master file

Source: Prof. Machida's high-level advice on the updated manuscript (skim-level
review, no point-by-point PDF this round). His verdict: **the technical content
and experiments are good, but the manuscript is not ready for submission because
of the presentation.**

This file is the standing prompt for every writing edit. Before and after each
batch of edits, check the work against PART 1 (rules) and update PART 2
(inventory) status.

---

## PART 0 — His verdict, in his own words

> "An academic paper does not need fashionable, cool, or executive summary
> sentences. It is not a business document nor a blog post for engineers."

> "Imagine the situation where you need to educate junior students to follow your
> study. You need to tell what you did very clearly in a logical order with
> moderate and introductory sentences using simple wording."

> "A good paper has a clear story with clear sentences, which does not mix many
> supplemental things. You do not need to decorate your study. Simply tell what
> you did, observed, and proposed."

> "We need very careful writing, sentence by sentence."

**Implication for how we work:** this is not a list of local find-and-replace
fixes. It is a prose rewrite. Every edit must be checkable against a rule below,
and no edit ships without the user's approval.

---

## PART 1 — The rules (check every edit against these)

### R1. Word choice must be simple, necessary, and consistent
- Do not introduce a technical term that is not commonly known unless it is
  **defined at first use** and then **used consistently everywhere**.
- If a term is unnecessary, delete it rather than define it.
- One concept = one name, for the entire manuscript. No synonyms for variety.
- No "fashionable" or business-flavoured vocabulary.

### R2. One message per sentence
- Do not pack two or more claims into one sentence using participles, trailing
  `, which ...`, or appended `, with the ... caveat ...` clauses.
- Split into two plain sentences instead.

### R3. Sentences must carry enough information to stand alone
- No rough, telegraphic sentences that assume the reader already knows the point.
- A junior student must be able to follow it without reconstructing context.

### R4. No ambiguous pronouns or high-context words
- Avoid `it`, `this`, `these`, `the two`, `that` where the referent is not
  immediately and unambiguously recoverable.
- Name the thing instead of pointing at it.

### R5. Section structure must be conceived before the prose
- Every (sub)section needs a clear purpose that its title announces.
- The reader must be guided through the proposal in a reasonable logical order.
- Supplemental material must not be mixed into the main story.

### R6. Standing style rules from earlier rounds (still in force)
- Use `$C_1$`, `$C_2$`, `$C_3$`, never bare `C1`.
- No em dashes. No `:` or `;` in prose. Headings use a colon, never a full stop.
- No `so` as a conjunction. No forward references that the reader must chase.

---

## PART 2 — Inventory of violation sites

Legend: **[S]** = sensei named this exact site. **[+]** = same pattern found by
sweeping the manuscript; he said "these are just samples."

Status column: `open` / `proposed` / `approved` / `done`.

---

### Group A — R1: inconsistent and questionable word choice

| # | Site | Current | Issue | Status |
|---|------|---------|-------|--------|
| A1 **[S]** | §II-A L189-190 | "The intermediate **signals** $z_1$ and $z_2$ are the **pipeline interfaces**, and each is a **cascade channel**." | Three unnecessary invented terms in one sentence. He says explain it more simply and drop them. | open |
| A2 **[S]** | §II-D title + L273-290 | "**Coupling Factor**", "the **terminal effect** $\Delta_3$" | "terminal" is an inappropriate word choice. Needs a plain replacement (e.g. planning metric / system output metric) applied everywhere. | open |
| A3 **[+]** | 20 further sites: L73, 203, 276, 285, 303, 345, 371, 589, 624, 626, 654, 664, 762, 880, 895, 898, 901, 978, 988, 998, 1113 | "terminal metric", "terminal effect", "terminal vector", "terminal readings", "two terminal clauses" | Same word he rejected, used throughout. Must be swept consistently, not fixed only where he pointed. | open |
| A4 **[S]** | L304, 345, 372, 380 | "**tolerance** $\tau_n$" | He reads it as a threshold. Rename to threshold everywhere, incl. Alg. 1 and the KwIn line. | open |
| A5 **[S]** | §IV-C L541 | "the standard nuScenes open-loop planning protocol used in the **end-to-end literature**" | Questionable phrasing. | open |
| A6 **[S]** | §V overview L588, §V-A L606 | "one **fully controlled** update" / "a single, **fully controlled** retraining" | Decorative qualifier. | open |
| A7 **[S]** | L76, 365, 430, 508, 594, 673, 688, 704 | "the **injected** effect" | Invented term used 8 times, never defined as a term. Either define once or replace with plain wording. | open |
| A8 **[S]** | CARA's key metric, named 5 different ways | L414 "interface-drift score", L417/837/879 "drift score", §III-B title "Interface-Drift **Front-End**", §V-G title "Drift **Screen**", L406-418 "the **screen**", L934 "the drift **proxy**", L1122 "front-end" | **Worst inconsistency in the paper.** He asks directly: "Is this the key metric used in CARA? If so, define it and keep using it consistently." One name must be chosen and used in every occurrence including figure axis label L854 and abstract L77. | open |
| A9 **[+]** | L327, 354-355, 382-384, 502-503, 507, 548, 569, 572, 611, 933, 1105 | "**profile**", "two-mode **profile**", "the two-mode **measurement**" | The same measurement has three names, and "profile" alone is used 11 times without ever being defined as a term. | open |
| A10 **[+]** | L288, 370, 574, 765, 783, 825 | "the entangled-enhancement **regime**" | Undefined term used as if standard, including as a table column heading. | open |
| A11 **[+]** | L189, 844 | "signals" (§II-A), "the front-end therefore **signals** the magnitude" | Same word used as noun-term and as ordinary verb, which compounds A1. | open |

---

### Group B — R2: two or more messages packed into one sentence

| # | Site | Current | Status |
|---|------|---------|--------|
| B1 **[S]** | §II-A last sentence, L192-193 | "When only $C_1$ is retrained, the output of a frozen downstream model changes only through its inherited input on this channel, never through its own parameters." | open |
| B2 **[S]** | §IV-C last sentence, L539-543 | "The L2 metric is the standard nuScenes open-loop planning protocol used in the end-to-end literature [...], with the differential-use caveat discussed in Sections V and VII." | open |
| B3 **[S]** | §V-E L764-767 | "Which updates land in the entangled-enhancement regime is not fully predictable from the training configuration, which is precisely why a downstream check is needed rather than a rule on the detector alone." | open |
| B4 **[S]** | §V-G L841-843 | "Separating L2-regressing updates from the rest is harder, and the score does not do it, with an AUROC of $\approx 0.48$." | open |
| B5 **[+]** | §V-A L616-619 | "The update therefore exhibits entangled enhancement in a driving pipeline, and it is a controlled, quantified instance of the correction-cascade pattern that the technical-debt literature describes only qualitatively." | open |
| B6 **[+]** | §VI-A L986-990 | "A regression on this metric may therefore mark a *safer* planner that scores worse on a measure rewarding ignored perception, and the terminal readings here should therefore be read as reduced trajectory agreement, not an established safety regression, and a closed-loop or NAVSIM-style metric is the appropriate safeguard." (three claims, two `and`s) | open |
| B7 **[+]** | §VII-B L1057-1061 | "Our use is differential (the same frozen planner, same scenes, one upstream change), and much of that weakness therefore cancels, and we report collision rate alongside L2 and revisit the caveat in Section VI." | open |
| B8 **[+]** | §VIII L1117-1120 | "Together these results show that model-level validation, and any single-metric check on the system output, is insufficient for safe maintenance of a machine learning system, since each of those six updates would have passed a review of $C_1$ alone and a review of the collision rate alone." | open |
| B9 **[+]** | §V-H L888-893 | "The best balanced operating point still catches every L2-regressing update (recall 1.0) but admits only two of nine non-regressing ones (specificity 0.22), at precision 0.46: the front-end can be tuned to catch every L2 regression, but at the cost of holding most other updates too." (also violates the no-colon rule) | open |
| B10 **[+]** | §III-A L366-368 | "CARA computes the change in its pipeline-mode metric $\Delta_k$ from (7) and reports the full vector $(\Delta_{i+1},\dots,\Delta_n)$, because adjacent entries can carry opposite signs that a single summary would hide." | open |

---

### Group C — R3: rough / short sentences that lack information

| # | Site | Current | Note | Status |
|---|------|---------|------|--------|
| C1 **[S]** | §IV-D L550-552 | "**Fine-tuning data size.** We use the full `singapore_train` set (the 229-scene Singapore split), a deterministic half of it, and a quarter. Smaller sets are the harder, more data-starved case." | He wrote "I could not understand." Needs to explain *why* smaller sets are studied and what "deterministic half" means. | open |
| C2 **[S]** | §IV-F last sentence, L574-575 | "Because $C_2$ and $C_3$ have no trainable parameters, the isolation check is exact." | Asserted without explaining the link. | open |
| C3 **[S]** | §V-A L619 | "Fig.~\ref{fig:cascade} shows the signature." | Tells the reader nothing. "The signature" is also undefined. | open |
| C4 **[S]** | §V-A L624 | "The two terminal metrics of this update moreover *diverge*." | Rough, plus "terminal" (A2) and an undefined use of "diverge". | open |
| C5 **[+]** | §V-C L673-674 | "Before any injected effect is interpreted, the frozen models must be shown to be untouched. The check validates the measurement setup rather than reporting a result." | Two terse assertions, no explanation of what the check is. | open |
| C6 **[+]** | §III-C L427-428 | "The method is therefore constructible only in a pipeline architecture." | Conclusion asserted without the intermediate reasoning. | open |
| C7 **[+]** | §V-D L703-704 | "The first downstream interface therefore dominates the effect injected by the retraining." | Rough summary sentence, plus A7. | open |

---

### Group D — R4: pronoun-heavy / high-context sentences

| # | Site | Current | Note | Status |
|---|------|---------|------|--------|
| D1 **[S]** | §II-B last sentence, L215-216 | "The improvement of the upstream model is what distinguishes **the two**, and **it** is the defining feature of the phenomenon." | He wrote "Very difficult to interpret." Both "the two" and "it" are unrecoverable. | open |
| D2 **[S]** | §II-C L249-250 | "**It** is what lets us attribute a change to the interface rather than to the model under test." | Sentence-initial unbound "It". | open |
| D3 **[S]** | §V-H L900-902 | "The value of the method is that **it** reports the full terminal vector and surfaces **the disagreement** for a human or a coordinated downstream retrain, rather than collapsing **it** into one admit/hold bit." | Two different referents for "it" in one sentence. | open |
| D4 **[+]** | §V-D L695-698 | "**This gap** conflates two sources [...]. **It** therefore bounds the combined effect rather than the perception error alone." | open |
| D5 **[+]** | §VI-A L971-975 | "Downstream models can **thus** absorb upstream error, and, symmetrically, upstream improvement can expose masked downstream sensitivity." | "symmetrically" + "masked" are high-context; the sentence carries the section's main claim. | open |
| D6 **[+]** | §V-H L894-895 | "The two terminal clauses of Algorithm 1 disagree here, **which is the point**." | Blog-style aside; "the point" is not stated. | open |
| D7 **[+]** | §II-C L262-264 | "Equation (6) is a testable property of the setup rather than an assumption. Any change in a frozen model's isolated-mode metric indicates contamination." | "contamination" undefined; the logical link is implicit. | open |

---

### Group E — R5: unclear subsection structure

| # | Site | Issue | Status |
|---|------|-------|--------|
| E1 **[S]** | §III-B "Interface-Drift Front-End" and §III-C "Why the Method Needs a Pipeline Architecture" | He wrote: "The intention of these two subsections is unclear by structure as well as section title. Maybe you did not conceive the section structure before writing." **This needs a structural decision, not a retitle.** §III-C in particular is an argument about applicability, not a component of the method. | open |
| E2 **[+]** | §V has 10 subsections | The overview paragraph (L586-602) announces "seven experiments" but the section then contains nine `\emph{Objective.}` subsections plus Training Diagnostics. The count and the structure do not match. | open |
| E3 **[+]** | §V-J "Training Diagnostics" L937-948 | Mixes three unrelated things: convergence curves, GPU cost, and a mechanism claim about ego-frame positions. Supplemental material mixed into the results story (R5, "does not mix many supplemental things"). | open |
| E4 **[+]** | §V-B title "The Detector Improves, Yet the System Degrades" | A findings-style title rather than a title announcing the subsection's purpose (which is ruling out an alternative explanation). | open |

---

## PART 3 — Open questions for the user (do not guess)

1. **A2/A3 replacement for "terminal"** — the word appears 21 times. Candidate
   replacements: "planning metric", "system-output metric", "final-stage metric".
   Needs one choice, applied everywhere.
2. **A8 single name for CARA's cheap metric** — candidates: "interface drift
   score" (define once, use always) vs something simpler. Also decides the §III-B
   and §V-G subsection titles and the Fig. 5 axis label.
3. **E1 structural decision on §III-B and §III-C** — retitle only, merge, move
   §III-C into the Discussion, or drop it.
4. **A7 "injected effect"** — keep as a defined term, or replace with plain
   wording throughout.

---

## PART 4 — Per-batch checklist

Run this before proposing any batch and again after applying it.

- [ ] Every changed sentence carries exactly one message (R2).
- [ ] Every pronoun in changed text has an unambiguous antecedent (R4).
- [ ] No new term introduced without a definition at first use (R1).
- [ ] Terms changed in this batch swept across the **whole** .tex, figures,
      tables, algorithm, and abstract (R1 consistency).
- [ ] `grep` clean for: em dash, prose `;`, prose `:`, `, so `, bare `C1/C2/C3`.
- [ ] Nothing in the batch contradicts `SENSEI_71_COMMENTS_AUDIT.md` (the 71
      earlier comments must stay neutralized).
- [ ] Numbers untouched unless the batch is explicitly about numbers.
- [ ] Build clean, page count still 10.
- [ ] `EXPERIMENT_LOG.md` updated.

---

## PART 5 — Progress log

| Date | Batch | Sites | Approved by user | Result |
|------|-------|-------|------------------|--------|
| 2026-08-17 | — | master file created, no edits yet | — | — |
