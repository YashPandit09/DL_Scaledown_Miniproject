# SpaceGen AI: Tasks and Development Plan

| Field | Value |
|---|---|
| Version | 1.6 (Week 3 built; what is left for the team is listed in the change notes, Section 12) |
| Team | [Member A] (ML and geometry lead), [Member B] (data, rules and app lead) |
| Duration | 21 days. "Day 1" is the day you start. |
| Companion docs | PRD.md, Tech_Spec.md, Architecture.md |
| Capacity assumption | About 4 focused hours per person per day, so 84 hours each and 168 person-hours in total. If you have less, cut P1 and P2 first, then use the cut list in Section 10. |

**Effort key:** S = 1.5 h, M = 3 h, L = 6 h (midpoints of the ranges; joint tasks are counted for both people).
**Priority key:** P0 = must, P1 = should, P2 = optional.

---

## 1. Plan overview

| Week | Theme | Exit gate |
|---|---|---|
| 1 (Days 1 to 7) | Foundation and data | **Gate 1 (Day 7):** rule checker tested, dataset v1 frozen, baseline numbers exist |
| 2 (Days 8 to 14) | Models and experiments | **Gate 2 (Day 14):** CVAE, evaluator and latent optimization work end to end; ablations E2 to E7 done; **configuration frozen** |
| 3 (Days 15 to 21) | Final runs, app, report, viva | **Final (Day 20):** everything frozen; Day 21 is buffer |

| Days | Focus |
|---|---|
| 1 to 2 | Setup, geometry, catalog, first rule checks |
| 3 to 4 | Reachability, quality score, rasterizer, dataset encoding, generator |
| 5 to 7 | Extra styles, perturbations, splits, dataset v1, baseline numbers, Gate 1 |
| 8 to 10 | CVAE, evaluator, latent optimization |
| 11 to 14 | Pipeline, ablations E2 to E7, first pass of E1 and E10, Gate 2 and configuration freeze |
| 15 to 17 | Final 3-seed runs, Streamlit app, figures, failure analysis, appendix |
| 18 to 20 | Report, README, viva preparation, final regeneration, demo rehearsal |
| 21 | Buffer and submission |

### 1.1 Effort budget (be honest about this)

| | Week 1 | Week 2 | Week 3 | Total | Capacity |
|---|---|---|---|---|---|
| P0, Member A | 25.5 h | 27 h | 28.5 h | 81 h | 84 h |
| P0, Member B | 27 h | 25.5 h | 28.5 h | 81 h | 84 h |
| **P0, both** | | | | **162 h** | **168 h** |
| P0 at the top of the effort ranges (S=2, M=4, L=8) | | | | 216 h | 168 h |
| P1 + P2 (conditional) | | | | 28.5 h | none left |

The P0 plan uses almost all of the available time. The buffer tasks (T48, and Day 21) are the only slack. **P1 and P2 tasks are conditional:** start one only if Gate 2 passed and you are ahead of schedule. If either of you has fewer than 4 focused hours per day, start cutting from Section 10 immediately. If your estimates run toward the top of the ranges, P0 alone needs about 216 h, which does not fit in 168 h: use the P0 reduction ladder in Section 10 before it becomes an emergency. The ladder cuts build, report and experiment work, closes the 48 h gap exactly, and leaves no slack.

---

## 2. Working agreements

- **Git:** `main` is always runnable. Work on short branches (`feat/geometry`, `feat/rules`, ...). The *other* member reviews every merge. This doubles as knowledge transfer for the viva.
- **Daily rhythm:** 10-minute sync at the start (yesterday, today, blockers) and a 10-minute merge check at the end.
- **Definition of done (any task):** code merged, tested where a test is listed, documented in the README or a docstring, and its output (table or figure) saved in `reports/`.
- **Cross-teaching:** Day 7, Day 14 and Day 18. Each member explains the *other's* modules and derives two formulas on paper.
- **Seeds and configs:** every run saves its config, seed and dataset hash. Never hard-code hyperparameters. The seed helper enables PyTorch deterministic mode (Tech Spec Section 9.2), so repeated runs on the same machine are bit-identical.
- **Compute:** keep datasets on the GPU as tensors (no `DataLoader`). The LOQ's GPU is the only GPU, so schedule it explicitly: the evaluator needs it (measured in T01 in deterministic mode: 21 s per epoch on the RTX 3050 6 GB vs 140 to 430 s on the CPU), while CVAE screening, latent optimization, rules and the app run fine on CPU on the second laptop. Launch long sweeps before leaving for the day.
- **Tasks:** every reproducible step is a task of `run.py` (`python run.py <task>`), because Windows has no `make`; where `make` exists, `make <task>` forwards to it. The `make figures` and `make all` below mean the matching `run.py` tasks.
- **Actual hours:** log the actual hours for every task. At Gate 1 and Gate 2, if actual hours are more than 15% above planned, start the reduction ladder (Section 10) at step 1 immediately.
- **Do not copy repository code.** Read papers and repos for ideas; write our own code and cite them.

---

## 3. Week 1: Foundation and data (Days 1 to 7)

| ID | Task | Owner | Pri | Effort | Depends | Done when |
|---|---|---|---|---|---|---|
| T01 | Create repo, folders, `requirements.txt` (include SciPy), `Makefile`, and the **seed helper with deterministic settings**; verify PyTorch runs; time one evaluator-sized epoch on each machine | A + B | P0 | S | none | `import torch` works on both laptops (GPU on the LOQ); two runs with the same seed give identical weights; folder structure matches Tech Spec |
| T02 | `geometry.py`: rotation to effective footprint, containment, exact overlap area, **penetration depth loss**, all in NumPy and PyTorch | A | P0 | M | T01 | Tests pass: analytic overlap, symmetry, NumPy equals PyTorch, **a fully contained box has non-zero loss gradient** |
| T03 | `catalog.yaml` and `catalog.py` (items, variants, prices, `rot_symmetry`, `group`, `needs_access`) | B | P0 | S | T01 | Loader returns items by slot; flags present |
| T04 | Draft `rules.yaml` (wall preferences, distance ranges, thresholds) and the assumptions list | B | P0 | S | T03 | All numeric assumptions in one commented file |
| T05 | `rules.py` hard checks H1 to H3 (in-room, overlap, door clearance) and the **door placement function** (wall orientation, centre formula) | B | P0 | M | T02, T04 | Hand-built rooms pass or fail as expected; door test passes on all 4 walls |
| T06 | `viz.py`: 2D layout plot (room, door, clearance, furniture with labels and facing arrows) | A | P0 | M | T02 | Any layout JSON renders to a PNG |
| T07 | `raster.py`: 4-channel 128x128 **fractional-coverage** raster on an 8 m canvas, built **on the fly on the GPU** as outer products of 1-D overlaps (no precomputed rasters: Set B would need about 16 GB); strips on all equivalent faces of symmetric items | A | P0 | M | T02 | Matches a brute-force supersampled reference; overlaps give values above single-item coverage |
| T08 | Reachability H4: distance transform, start at the **door clearance centre**, `scipy.ndimage.label`, access points on **every equivalent face** for items with `rot_symmetry > 1` | B | P0 | M | T05 | Empty room passes; blocked door and walled-off item fail; start cell passable; a side table against the north wall passes |
| T09 | Quality score `S` (alignment, relations, circulation, space); orientation terms skip items with `rot_symmetry > 1` | B | P0 | M | T08 | Score in [0,1]; sensible on 5 hand-made good and bad layouts |
| T10 | `dataset.py`: encode `x` and `c`, normalization, presence masks, **canonicalization**, tensors on GPU | A | P0 | M | T02, T03 | Round-trip encode then decode reproduces the canonical layout |
| T11 | `generator.py`: living-room styles (a), (b) and (c) with jitter, **area-dependent item subsets**, canonical labels, rejection through the checker; log rejection by area bin and item count. *v1.4: styles (b) and (c) moved here from T13, because style (a) cannot keep the sofa-TV distance in range in about a quarter of rooms* | B | P0 | L | T05, T08 | 100% of Set A layouts pass the checker; rejection plot exists; canonicalization test passes |
| T12 | `baselines.py`: B1 uniform random; B2 statistical sampler; G0 wrapper around the generator | A | P0 | M | T10, T11 | All three return layouts for any condition vector |
| T13 | Set B perturbations including **near-miss**; perturbation type stored per sample (*v1.4: the styles were built in T11*) | B | P0 | M | T11 | Set B roughly balanced valid vs invalid; per-type counts logged |
| T14 | Splits: 70/15/15, plus separately sampled interpolation, unseen-combination, out-of-range and G0 reference sets | A | P0 | S | T10 | Split files saved; no leakage (asserted in a test) |
| T15 | Generate dataset v1 (Set A about 30k, Set B about 60k); metadata and hash; histograms; **calibrate `f_max`** from the acceptance rate | B | P0 | M | T13, T14 | `data/v1/` complete; histograms of room area, item counts, valid ratio, rejection rate saved |
| T16 | Baseline evaluation harness (build on a 2k-sample mini dataset earlier), then run B1, B2, G0 on dataset v1: RVR, overlap, reachability, quality, diversity, cost per valid layout | A | P0 | M | T12, T15 | `reports/tables/baselines.csv` exists |
| T17 | Notebook `01_manual_backprop.ipynb`: toy 2-item network, hand-derived gradients equal autograd | A | P0 | M | T01 | Difference below 1e-6; derivation in markdown cells |
| T18 | **Gate 1 review and cross-teaching #1** | A + B | P0 | S | T01 to T17 | Checklist in Section 7 passes |

**Day-by-day (Week 1)**

| Day | A | B |
|---|---|---|
| 1 | T01 (together), T02 | T01 (together), T03, T04 |
| 2 | T06 | T05 |
| 3 | T07, T10 (start) | T08, T09 (start) |
| 4 | T10 (finish), T12 (start) | T09 (finish), T11 (start) |
| 5 | T12 (finish), T14, harness for T16 on the mini dataset | T11 (finish), T13 |
| 6 | T17 | T15 (dataset v1 must be finished by the end of Day 6) |
| 7 | T16 (final numbers), T18 | T18; fix any generator issues |

**As built (v1.4):** the tasks ran in the order T01 to T11, T13, T14, T15, T12, T16, T17, because B2 is fitted on the training split of dataset v1. The harness for T16 was tested on a tiny dataset built inside the tests.

**Gate 1 fallbacks (if behind on Day 7):** drop generator style (c); postpone Set B perturbation types (iii) and (iv); keep reachability but simplify access points. Pinned furniture, real rooms, bedroom and 3D wait until Gate 2 passes comfortably. *(Not needed in v1.4: all styles, perturbation types and access checks were built.)*

---

## 4. Week 2: Models and experiments (Days 8 to 14)

| ID | Task | Owner | Pri | Effort | Depends | Done when |
|---|---|---|---|---|---|---|
| T19 | `models/cvae.py`: encoder, decoder with **logit** rotation head, masked loss with `F.cross_entropy`, KL; tiny-batch overfit test | A | P0 | L | T10 | Loss on 64 samples goes near zero in a few hundred steps |
| T20 | `models/evaluator.py`: 4-conv CNN and training loop (BCE plus regression) | B | P0 | L | T07, T15 | One epoch trains; loss decreases |
| T21 | `train_cvae.py`: KL annealing, **early stopping at `beta_target` after annealing**, CSV logging (loss, per-layer gradient norms, dead units, active units) | A | P0 | M | T19 | Checkpoint saved with config and dataset hash; test shows early stopping cannot fire during annealing |
| T22 | Train the evaluator (**E9a**); precision, recall, F1, AUC, confusion matrix, **F1 per perturbation type**, Spearman with the rule score | B | P0 | M | T20 | `reports/tables/evaluator.csv`; per-type F1 table; confusion matrix figure |
| T23 | `latent_opt.py`: penetration-depth loss, out-of-room and door terms, **anchor to `z0`**, pinned term, optional surrogate hook | A | P0 | L | T21, T02 | Final loss below initial loss for at least 90% of candidates; finite-difference gradient test passes including containment |
| T24 | `models/mlp_baseline.py`: hand-crafted features and MLP (**E9b**) | B | P1 | M | T22 | CNN vs MLP table |
| T25 | `pipeline.py`: feasibility, variant selection, sampling, checks, ranking, top-3 with diversity | B | P0 | L | T21, T22, T23 | End-to-end call returns 3 layouts or a clear message; smoke test passes |
| T26 | **E2** (losses, one factor at a time, clean vs outlier), **E3a** (six activations, BatchNorm **on**, depth 2; decides the deployed activation) and **E3b** (BatchNorm-off diagnostic, depth 2 and 6, with BN-on references), 1-seed screening | A | P0 | L | T21 | Loss and gradient-vs-error plots; E3a table and decision; gradient-norm plot; dead-unit chart; BN-on vs BN-off comparison |
| T27 | **E5** (beta and latent size, one factor at a time) | A | P0 | S | T26 | Trade-off plot |
| T27b | **E4** (optimizers) and **E6** (BatchNorm, dropout, early stopping) | B | P0 | M | T21 | Convergence curves; regularization table |
| T28 | **E1 first pass** on the default configuration: B1, B2, G0, M1, M2 including cost per valid layout | B | P0 | M | T25, T16 | Debug run of the main table (not the final numbers) |
| T29 | **E7** (Sigmoid vs Linear + clamp) and **E8** (optimization steps) | A | P0 | M | T23, T26 | Wall-proximity error chart; RVR vs steps curve |
| T30 | **E10 first pass** on all four test sets | B | P0 | M | T25, T14 | Debug run of the grouped bar chart |
| T31 | E9b write-up: CNN vs feature MLP, with the caveat that the features encode the rules | B | P1 | S | T24 | Paragraph plus table |
| T32 | Bedroom config and generator; retrain CVAE with the same code | B | P2 | L | Gate 2 pass | Bedroom RVR table for M1 and M2 |
| T33 | **Gate 2 review, cross-teaching #2 and configuration freeze** (write `configs/frozen.yaml` from E2 to E7). *v1.5: the screening shortlists; three seeds of the shortlisted settings are compared with M2 on Set A validation rooms (`python run.py gate2`), because the screening's M1 check sampled test rooms* | A + B | P0 | S | T19 to T30 | Checklist in Section 7 passes |

**Day-by-day (Week 2)**

| Day | A | B |
|---|---|---|
| 8 | T19 | T20 |
| 9 | T19 (finish), T21 (start) | T20 (finish), T22 (start) |
| 10 | T21 (finish), T23 (start) | T22 (finish), T25 (start) |
| 11 | T23 (finish), T26 (start; launch screening runs) | T25 (continue) |
| 12 | T26 (finish), T27, T29 (start) | T25 (finish), T27b |
| 13 | T29 (finish) | T28, T30 |
| 14 | T33 | T33; **launch final 3-seed runs on the frozen configuration before leaving** |

**Training budget note.** With one factor at a time and 1-seed screening, E2 to E7 need roughly 65 short CVAE trainings (about 47 screening runs plus 3-seed confirmations of the default and the best settings). Keep data on the GPU as tensors and run sweeps overnight. The CVAE is small enough for CPU, so sweeps can run on the second laptop while the LOQ's GPU trains the evaluator.

**Ordering rule.** E1, E8 and E10 headline numbers are produced only on the **frozen** configuration (T33b). The first passes in T28 and T30 exist to debug the pipeline.

**Gate 2 fallbacks:**
- If the CVAE is unstable by Day 10: reduce `K` to 4 slots, lower `beta_target`, use 1 hidden layer, and skip dropout sweeps in E6.
- If latent optimization does not help by Day 12: keep M1 as the main method and report latent optimization as a partial or negative result, explained honestly.
- If the evaluator is weak: report it truthfully, rank by rule score, and keep the CNN as an experiment.

---

## 5. Week 3: Final runs, app, report, viva (Days 15 to 21)

| ID | Task | Owner | Pri | Effort | Depends | Done when |
|---|---|---|---|---|---|---|
| T33b | **Final runs on the frozen configuration:** 3-seed training, then E1, E8 and E10 headline numbers | B | P0 | M | T33 | Final tables with mean and standard deviation saved |
| T34 | Streamlit app v1: inputs, feasibility message, top-3 view with metrics | B | P0 | L | T25 | Full flow works from a clean start |
| T35 | `experiments/make_figures.py`: rebuild all tables and figures from saved CSV logs | A | P0 | M | T26 to T30, T33b | `make figures` regenerates everything |
| T36 | App: "Compare methods" tab and JSON or PNG export | B | P0 | M | T34 | B1, B2, G0, M1, M2 shown on the same input |
| T37 | Failure-case analysis (at least 10 cases, classified with causes) | A | P0 | M | T33b | Gallery figure plus written discussion |
| T38 | **Pinned furniture** (position and optional facing; rotation overwritten, position snapped) in pipeline and UI; build the **G0-pin** baseline; experiment E12 split by pinned item. *Top priority among P1 items.* | A | P1 | L | T23, T34 | Validity after snapping, cost per valid layout and pre-snap displacement reported per pinned item against G0-pin, B1 and B2 |
| T39 | Real-room test set (15 to 20 rooms) and E11 | B | P1 | M | T25 | Gallery and RVR reported |
| T40 | 3D box view from layout JSON (Plotly) | B | P2 | M | T34 | Rotatable 3D view in the app |
| T41 | Math appendix: clean write-up of Tech Spec Section 8 (already drafted) | A | P0 | S | none | Reviewed by B |
| T42a | Report draft, part A: methods, derivations, experiments, results | A | P0 | L | T35, T37 | Sections 4 and 5 of the outline complete with figures |
| T42b | Report draft, part B: introduction, related work, dataset, rules, app, demo | B | P0 | L | T34 | Sections 1 to 3 and 6 complete |
| T43 | Limitations, ethics and deployment section | A | P0 | S | T42a | Uses the points in Tech Spec Section 10 |
| T44 | README and fresh-clone check: the smoke test passes, and regenerating dataset v1 with the same seed reproduces the dataset hash (start it on Day 17) | A | P0 | M | T35 | Fresh clone runs the smoke test; dataset hash matches |
| T45 | Viva preparation: derivations on paper, one-page cheat sheet, prepared answers to Section 8 | A + B | P0 | M | T41 | Both members can answer every question without notes |
| T46 | **Launch the full regeneration overnight on Day 19** (`make all`: datasets, all trainings, all experiments; expect many unattended hours, and if it cannot finish in one night regenerate only E1, E8, E10 and the figures). On Day 20 compare with the saved results: on the same machine, with the deterministic settings, they must be **exactly equal**; on another machine use the tolerance in Tech Spec Section 9.2. Tag `final`. | A + B | P0 | S | T44, T45 | Exact equality on the same machine (or the stated tolerance across machines); git tag `final` |
| T47 | Demo rehearsal and mock viva (twice, timed): 5-minute demo script and backup screenshots | A + B | P0 | M | T36, T45 | Demo runs without surprises; each member answers questions on the other's modules |
| T48 | Buffer, fixes, submission | A + B | P0 | M | T46 | Submission packaged |
| T50 | Differentiable soft raster, surrogate-guided latent optimization (M3) and **E8b** | A | P1 | L | Gate 2 pass, T23 | Table comparing M2 and M3, including signs of surrogate exploitation |

**Day-by-day (Week 3)**

| Day | A | B |
|---|---|---|
| 15 | T35 (start with the final runs' output) | T33b (collect final runs), T34 (start) |
| 16 | T37, T38 (P1) | T34 (finish), T36 |
| 17 | T41, T44 | T42b (start) or T39 (P1) |
| 18 | T42a, T43; cross-teaching #3 | T42b |
| 19 | T45; **launch the T46 overnight run** | T45 |
| 20 | T46 (check headline numbers, tag), T47 | T46 (check), T47 |
| 21 | T48 (buffer) | T48 (buffer) |

T50 and T40 have no fixed day: do them only in gaps and only if Gate 2 was comfortable.

---

## 6. Report outline (suggested)

1. Introduction and problem statement, including why a learned model when rules exist (PRD Section 1.4)
2. Related work (LayoutVAE, constrained layout generation with latent optimization, ATISS, others) and an honest gap statement
3. Dataset: generator, rules and assumptions, canonicalization, statistics, splits, rejection-rate plot
4. Methods: CVAE (with derivations), CNN evaluator (with its honest role), latent optimization, baselines and the G0 reference
5. Experiments E1 to E12: setup, results, analysis
6. Demo (screenshots of the app)
7. Failure cases
8. Limitations, ethics and deployment
9. Conclusion and future work
10. Appendix: math derivations, hyperparameters, reproducibility instructions, references

---

## 7. Checkpoint checklists

**Gate 1 (Day 7)** (evidence for each box in `reports/gate1.md`)
- [x] Geometry (including containment gradient), rules, door placement and rasterizer tests pass
- [x] Reachability passes on an empty room and the start cell is passable
- [x] Every Set A layout passes the checker and is canonical (asserted)
- [x] Dataset v1 generated, hashed, split, with separately sampled held-out sets
- [x] Histograms and the rejection-rate plot saved; `f_max` calibrated
- [x] B1, B2 and G0 baseline numbers saved (including cost per valid layout)
- [x] Manual backprop notebook matches autograd
- [ ] Each member can explain the other's Week 1 modules
- [ ] Actual hours logged for Week 1 tasks; if more than 15% over planned, start the reduction ladder now

**Gate 2 (Day 14)** (evidence for each box in `reports/gate2.md`)
- [x] CVAE trains without collapse (active units above 0); early stopping test passes
- [x] Evaluator F1 reported overall and per perturbation type
- [x] Latent optimization lowers the constraint loss for at least 90% of candidates, and its effect on RVR and diversity is measured (either direction, reported honestly)
- [x] Pipeline returns 3 layouts end to end
- [x] E2 to E7 have screening results saved; the deployed activation is chosen from E3a (BatchNorm on); the best configuration is written to `configs/frozen.yaml`
- [x] First passes of E1 and E10 run without errors
- [x] Final 3-seed runs on the frozen configuration launched
- [ ] Actual hours logged for Week 2 tasks; if more than 15% over planned, apply the reduction ladder before starting any P1 or P2 task
- [x] Go or no-go recorded for: pinned furniture (T38), real rooms (T39), surrogate (T50), 3D (T40), bedroom (T32) (`reports/gate2.md`: T38 built; T39 stays open until the team has measured rooms; the others no-go)

**Go/no-go rule for extras:** add an extra only if all Gate 2 boxes are ticked *and* the remaining days cover the report and viva preparation with at least one spare day.

**Final (Day 20)** (evidence in `reports/regeneration.md`)
- [x] Fresh-clone reproduction works
- [x] Full regeneration into an empty folder compared with the saved results: exactly equal (9 October 2026)
- [x] All figures regenerate from `make figures`
- [x] Report complete with appendix
- [ ] Demo and mock viva rehearsed twice
- [x] Git tag `final` created

---

## 8. Viva preparation

### 8.1 How to prepare
1. Each member owns a set of derivations (suggested: A owns KL, reparameterization, Adam and the overlap gradient; B owns the BCE and softmax gradients, precision and recall, and reachability logic).
2. Each member must explain every module at block-diagram level, and their own modules line by line.
3. Run a mock viva on Day 19 to 20: one member asks, the other answers on paper or a whiteboard, then swap.
4. Prepare one figure per experiment to point at while answering.

### 8.2 Likely question areas (use as drills)
1. Why a CVAE instead of a plain MLP that maps conditions to one layout?
2. **Your generator already makes 100% valid layouts, so why train a network?** (be honest: G0 may be cheaper per valid layout for unpinned requests; the case rests on pinned completion against G0-pin, amortized sampling and differentiable repair; have the E1 and E12 numbers ready, whichever way they fall)
3. **The rule score is exact, so what does the CNN evaluator add?** (honest answer: not needed for correctness and not faster than the rule check; it is a CNN study (E9a), a possible differentiable surrogate (E8b), and a learned ranker reported next to the exact score)
4. Derive the KL term. What is the reparameterization trick and why is it needed?
5. What is posterior collapse? How did you detect it? (active units)
6. Why Sigmoid on positions? What happens near 0 and 1? (E7)
7. Why Softmax with cross-entropy for rotation? Derive the gradient `p - y`. Why logits with `F.cross_entropy`?
8. MSE vs MAE vs Huber: gradients and outlier behaviour. What did E2 show?
9. What did E3 show about vanishing gradients and dying ReLU? Why did you turn BatchNorm off for it?
10. What does BatchNorm do during training and at inference? Why `eval()` during latent optimization?
11. What does dropout do? What did E6 show about the train-validation gap?
12. Compare SGD, momentum, RMSProp and Adam. Write the Adam update. What did E4 show?
13. Why is latent optimization not cheating? (validity is checked by rules; anchor to `z0`)
14. Why penetration depth and not intersection area as the loss? What happens to a contained box?
15. Why an anchor to `z0` and not a `||z||^2` prior?
16. Why does the loss use a presence mask? (Hadamard product)
17. Why divide coordinates by room size? (scaling and generalization)
18. Why did you canonicalize rotations and swap nightstands? What would go wrong otherwise?
19. Precision vs recall for "valid": which error is worse here? What is your near-miss F1?
20. Show the matrix shapes through the decoder for a batch.
21. What happens when the room is bigger than anything in training? Explain the difference between the unseen-combination and out-of-range sets. (E10)
22. Your data is synthetic and rejection sampling skews it: what are the consequences and how did you reduce them?
23. Where is early stopping used, and why does it start after annealing?
24. What would you do with more time or real data?
25. Why does the checker treat every face of a side table alike, and what breaks if it does not? (canonical labels vs single-face access checks)

### 8.3 Cheat-sheet contents (one page each)
Formulas (Tech Spec Section 8), architecture diagrams, results table (E1), list of assumptions, list of limitations.

---

## 9. Risk triggers and responses

| Trigger | When to check | Response |
|---|---|---|
| Generator produces many rejected layouts, or crowded small rooms vanish from the data | Day 5 | Reduce item subsets for small rooms, loosen jitter, calibrate `f_max`; state the skew in the report |
| Gate 1 missed | Day 7 | Cancel all P1 and P2; use Day 8 as catch-up and shift Week 2 by one day |
| CVAE outputs nearly identical layouts | Day 9 to 10 | Lower `beta_target`, check active units, confirm annealing and early-stopping logic; reduce hidden size |
| Latent optimization increases overlap or hurts diversity | Day 12 | Lower `lr`, raise the anchor weight, cap steps; report the trade-off |
| Evaluator accuracy high but near-miss F1 low | Day 12 | Report it; the checker stays the authority; do not oversell the CNN |
| Sweeps take too long | Day 11 to 13 | Cut to 1 seed for screening, fewer epochs, drop the dropout sweep |
| GPU problems | Any | Switch to CPU (models are small); reduce batch size |
| One member unavailable for 2 or more days | Any | The other continues P0 only; cancel P1 and P2; share notes daily |
| Report behind schedule | Day 17 | Freeze experiments; only fixes allowed |

---

## 10. Cut list (in the order to drop things if time runs short)

1. 3D view (T40)
2. CNN surrogate and E8b (T50)
3. Real-room test set (T39)
4. Bedroom domain (T32)
5. Feature-MLP comparison (T24, T31)
6. E6 dropout sweep and extra learning rates in E4 (keep at least Adam vs SGD)
7. Generator styles beyond (a) (already built in T11, so nothing left to cut here)
8. Pinned furniture (T38) is the **last** extra to drop, because it is the strongest argument for the learned approach

Never cut: the checker, dataset v1, baselines including G0, CVAE, evaluator, E1, E2, E3a, the report appendix, and viva preparation.

### P0 reduction ladder (if effort runs at the high end)

At the top of the effort ranges P0 needs 216 h against 168 h, a 48 h gap. The experiment tasks alone (T26 to T30, T27b, T33b) are only 30 h at that level, so deleting all of them would still leave 186 h. The ladder therefore cuts build, report and experiment work, cheapest regret first. Savings are person-hours at top-of-range effort (S=2, M=4, L=8). Each step weakens something specific, and the report must say so.

| Step | Cut | Tasks | Saves | Cumulative |
|---|---|---|---|---|
| 1 | Write the report from the Tech Spec text (methods, dataset, ethics, appendix); fold the appendix and ethics write-ups into the report | T42a L to M, T42b L to M, drop T41 and T43 | 12 h | 12 h |
| 2 | Minimal README; one rehearsal with the mock viva inside T45 | T44 M to S, T47 M to S | 6 h | 18 h |
| 3 | Drop the app's Compare tab and export; show the E1 figure in a Results tab | T36 | 4 h | 22 h |
| 4 | Quality-score relations: sofa to TV only | T09 M to S | 2 h | 24 h |
| 5 | Three Set B perturbation types instead of five (per-type F1 becomes coarser) | T13 M to S | 2 h | 26 h |
| 6 | Core figures only; 5 failure cases instead of 10 | T35, T37 M to S | 4 h | 30 h |
| 7 | Minimal app: inputs, feasibility message, top-3 view | T34 L to M | 4 h | 34 h |
| 8 | Experiments: 1-seed screening for E2 to E7, drop E7, E4 to Adam vs SGD, E6 to BatchNorm on/off, E10 on two test sets, E1 headline with 2 seeds (never 1) | T26 L to M; T27b, T29, T30 M to S | 10 h | 44 h |
| 9 | Minimal 2D plot; one-layer backprop notebook | T06, T17 M to S | 4 h | 48 h |

Applying the whole ladder closes the gap exactly and leaves **no slack**, so treat steps 7 to 9 as a last resort. Decide from logged actual hours at Gate 1 and Gate 2, not from a feeling in Week 3.

---

## 11. Submission checklist

- [x] Code repository tagged `final` (move the tag if you commit again before submitting: `git tag -f final && git push -f origin final`)
- [x] README with setup and reproduction commands
- [x] Dataset generator, config and seed (the dataset is rebuilt from them with the identical hash; its files are not committed)
- [x] Trained checkpoints and `configs/frozen.yaml`
- [x] Results tables and figures
- [ ] Report (with math appendix, limitations, ethics, references): written, and built as `reports/pdf/report.pdf`. **Put your names in line 3 of `reports/report.md`, then run `python run.py pdf` again**
- [x] Demo app and screenshots as backup (`reports/demo/app_*.png`)
- [ ] Viva cheat sheet printed or on a phone: the sheets are the last pages of `reports/pdf/viva_prep.pdf`

---

## 12. Change notes

**v1.6 (Week 3 built)**
- Built: T33b (final runs), T34 and T36 (app), T35 (figures), T37 (failure analysis), T38 (pinned furniture and E12, the one extra taken), T41 (math appendix), T42a, T42b and T43 (report, `reports/report.md`), T44 and T46 (the fresh-clone check and the full regeneration into an empty folder, which reproduced every table, every trained model and the dataset exactly; log in `reports/regeneration.md`), `python run.py pdf` (the documents as PDF files, for T48), the documents for T45 (`reports/viva_prep.md`) and T47 (`reports/demo_script.md`), and the backup screenshots of the app (`reports/demo/app_*.png`).
- **Left for the team, because only people can do them:** the names in the report, the Week 2 hours (Gate 2 box 8), the decision on T39 if rooms can be measured, cross-teaching sessions #2 and #3, learning the viva answers (T45), rehearsing the demo and the mock viva twice (T47), and the submission itself (T48).
- Not built: T39 (real rooms: it needs 15 to 20 rooms measured by the team), T50 (surrogate), T40 (3D view), T32 (bedroom), T24 and T31 (feature MLP).
- `run.py` gained the tasks `e8`, `e12`, `failures`, `figures`, `all`, `check-regeneration`, `demo-assets` and `app`.
- Timings are measured with the methods taking turns (Tech Spec v1.6): the laptop's speed changes with its power and thermal state. Long runs need mains power.

**v1.5 (Week 2 built, Gate 2 review)**
- Week 2 ran in the order T19 and T20, T21 and T22, T23 and T25, T26, then T27, T27b and T29 together, T28 and T30, T33. The P1 tasks T24 and T31 and the P2 task T32 were not started.
- T33 compares three seeds of the shortlisted settings with M2 on Set A validation rooms (`python run.py gate2`). The screening's M1 check sampled test rooms, so it can only shortlist, and M1 turned out to be a poor guide: Tanh nearly doubles M1's validity but lowers M2's.
- Gate 2: six of nine boxes ticked with evidence in `reports/gate2.md`. The final 3-seed runs (T33b), the Week 2 hours and the go or no-go for the extras remain.
- New `run.py` tasks: `screen`, `e1`, `e10`, `gate2`.
- Technical decisions taken while building (latent-optimization stopping, experiment details, the E1 and E10 additions, the frozen configuration) are in the Tech Spec v1.5 change log.

**v1.4 (Week 1 built, Gate 1 review)**
- T11 builds all three generator styles; T13 keeps only the Set B perturbations. Style (a) alone cannot keep the 1.5 to 3.5 m sofa-TV distance in about a quarter of rooms (Tech Spec 2.4).
- Week 1 ran in the order T01 to T11, T13, T14, T15, T12, T16, T17: B2 is fitted on the training split of dataset v1.
- Gate 1: seven of nine boxes ticked with evidence in `reports/gate1.md`; the cross-teaching session and the hours log remain.
- Compute measured in T01 (21 s per evaluator epoch on the GPU, 140 to 430 s on the CPU) replaces the reviewer's figures.
- `run.py` is the task runner; `make` forwards to it where it exists.
- Technical decisions taken while building (reachability access line, near-miss range, `f_max = 0.38`, baseline details) are in the Tech Spec v1.4 change log.
