# Sensei (Machida) — all 71 comments, image-by-image audit

Method: every comment extracted from `AIoT2026-ifty-v1-comment.pdf` with its page,
column and y-position, then each one **read from a rendered image crop of the
annotated page** (strikeouts carry no text, so position-only inference is unsafe)
and verified against the current `IEEE_Conference_Paper_E1_v4.tex`.

**RESULT: 71/71 neutralized. Build clean, 10 pages, 0 errors, 0 undefined refs.**

## Page 1 (title, abstract, keywords, intro start)
| # | Comment | Fix |
|---|---------|-----|
| 1-3 | Carets "Tackling" / "the Model Pipeline for" / "Vehicles" | Title = "Tackling Entangled Enhancement in the Model Pipeline for Autonomous Vehicles" |
| 4 | more objective, cite NeurIPS2021 self-defeating, [4], [13]; gap = entire AV pipeline; prior work limited to two sequential models | Intro rewritten; 3 citation placeholders added; "two sequential models" + "entire perception-prediction-planning pipeline" stated |
| 5 | abstract "rewritten after reorganizing" | Abstract rebuilt around the 3 contributions |
| 6 | colloquial -> "To address this gap, this paper proposes..." | Paragraph 3 opens with it |
| 7 | avoid ":" and ";" | 0 in intro |
| 8 | "Yolo is not a camera detector" | "an object detector applied to the front-camera image" |
| 9 | no C1/C2/C3 in introduction | 0 in intro |
| 10 | "down stream models" | "the downstream models were tuned to..." |
| 11 | contributions threefold (formal def / evidence / CARA) | exactly those 3 |
| 12 | keywords alphabetical | reordered |

## Page 2 (findings list, contributions, Fig.1, EE definition)
| # | Comment | Fix |
|---|---------|-----|
| 13 | F1-F7 too detailed for intro | list removed from intro |
| 14 | use $C_1$ not C1 in entire manuscript | 0 bare C1/C2/C3 left (incl. TikZ axis labels) |
| 15 | "Remove this because of mismatch" (Fig.1 + its sentence) | Fig.1 pipeline diagram and sentence deleted |
| 16 | contributions reformulated; drop "dual-mode measurement"; "pipeline" not "modular" | done; 0 "modular", 0 "dual-mode" |
| 17 | "Entangle enhancement occurs when" | definition opens with it |
| 18 | "[3]. This entanglement happens" | cited + reworded |
| 19 | what does "strict" mean | removed from the definition |

## Page 3 (formal model, notation)
| # | Comment | Fix |
|---|---------|-----|
| 20 | "so" as conjunction | 0 in document |
| 21 | "Not formal" (decoupling passage) | passage removed |
| 22 | unclear; omit; clarify coupling factor as the formal definition | synthetic check removed, rho defined as EE coupling |
| 23,24 | carets ". In" / "empirically show" | superseded (passage removed) |
| 25 | "unclear" (heading "Dual-Mode Evaluation") | renamed "Cascade Gap and the Isolation Property" |
| 26 | these belong in Section IV | Retraining Procedure + Metrics moved to Sec IV |
| 27 | reposition; GT not always available | reframed; "in deployment, where ground truth is not observed" |
| 28,29 | carets "direct" / "the" | "the direct output of the upstream neighbor" |
| 30 | optional extra formalization | skipped (user decision) |
| 31,32,33 | "Let Mk be an" / "for zk" / "metrics for the isolated and pipeline modes" | `$M_k(z_k)$` operator introduced, modes named |
| 34 | Gamma_k = Mk(z^pipe) - Mk(z^iso) | exact match |
| 35 | use z_k'; isolation = M_k(z^iso)=M_k(z^iso') | exact match |
| 36 | "?" on "apparatus" | -> "measurement setup" |
| 37 | "?" on "bit-exactly" | removed |
| 38 | Delta_k = Mk(z^pipe') - Mk(z^pipe) | exact match |
| 39 | "This is not defined" (Delta_3^iso) | equation removed |

## Page 4 (CARA)
| # | Comment | Fix |
|---|---------|-----|
| 40 | Sec II gives the formal definition | III opens "The formal definition of Section II..." |
| 41 | avoid ":" ";" | 0 |
| 42 | avoid "so" | 0 |
| 43 | critical section, written roughly | rewritten as 3 explicit steps |
| 44 | write it more formally | inputs/outputs + numbered procedure |
| 45,46 | "updated" / "original" (figure boxes) | figure relabelled |
| 47 | "Need more information" (audit set) | audit set now defined (fixed scene collection, GT-annotated, held constant) |
| 48 | drop R, use C_i / C_i' | R removed |
| 49 | "reconsider the term" (dual-mode profile) | -> "two-mode profile"; 0 "dual-mode" in document |
| 50 | caret "original" | "original $C_i$" |
| 51 | write carefully; no ambiguous "it"; **do not rely on later sections** | rewritten; 0 forward refs to Sec V |
| 52 | need a clear CARA procedure tied to Fig2/Alg1 | "three steps, shown in Fig.2 and stated in Alg.1" |
| 53 | "explain details" (eight driving classes) | all eight classes named |
| 54,55 | "We employ" / "as a perception model (C1)" | "We employ YOLOv11n as the perception model $C_1$" |

## Page 5 (setup end, Section V start)
| # | Comment | Fix |
|---|---------|-----|
| 56 | strikeout of the LiDAR/absolute-magnitude clause | whole clause deleted (CenterPoint cite gone) |
| 57 | "To generate the updated model, the YOLOv11n is" | sentence split exactly as suggested |
| 58 | caret "effect." | overview rewritten |
| 59 | "What is the adopted model?" | "we adopt a constant-velocity motion model" |
| 60 | "is used for a planning model C3" | "The Intelligent Driver Model is used for the planning model $C_3$" |
| 61 | motivation entirely unclear from the top | new overview: 2 goals + objective of each of 6 experiments |
| 62 | "Need more careful description. I could not understand." | planner described step by step (grid -> score -> select) |
| 63 | caret "reflects" | sentence removed in restructure |
| 64 | "not encourage style" | PDM sentence ";" removed |
| 65 | strikeout "original" (incumbent) | 0 "incumbent" in document |
| 66 | duplicate of Table 1 -> show either one | before/after bar chart deleted, Table kept |
| 67 | "fine-tuned C1'" | "only the Singapore fine-tuned $C_1'$ changes" |
| 68 | "Need to be more precise. How?" | states the fixed seed + deterministic $C_2$/$C_3$ |
| 69 | "?" on "five-quantity profile" | term removed |
| 70 | metric is metric; "isolated/pipeline metrics" wrong | "the metrics of $C_2$ and $C_3$, each evaluated twice, once in isolated mode and once in pipeline mode" |
| 71 | caret "C1 and C1'" | "for both $C_1$ and $C_1'$" |

## Whole-document rule check (all zero)
`, so ` 0 · `modular` 0 · `incumbent` 0 · `camera detector` 0 · `dual-mode` 0 ·
`apparatus` 0 · `bit-exact` 0 · F-tags 0 · bare C1/C2/C3 0 · CenterPoint 0 ·
prose semicolons 0 · prose colons 0 · em dashes 0 (only a LaTeX comment).

## Still open (NOT comment items)
- The 3 citation placeholders need real .bib entries from you:
  `TODO_neurips2021_selfdefeating`, `TODO_sentence_classification`, `TODO_context13`.
- The struck LiDAR/absolute-magnitude caveat was a limitation; it belongs in a
  limitations section, which the paper still does not have.
