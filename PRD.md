# SpaceGen AI: Product Requirements Document (PRD)

| Field | Value |
|---|---|
| Project | SpaceGen AI: constraint-aware interior layout generation with deep learning |
| Context | Deep Learning mini project (100 marks) with viva |
| Team | 2 members ([Member A], [Member B]) |
| Duration | 3 weeks (21 days) |
| Hardware | Lenovo LOQ laptop with an NVIDIA RTX 3050 (VRAM to be confirmed); the second laptop is assumed CPU-only. The CVAE, latent optimization, rules and app run on CPU; the evaluator needs the GPU (see NFR-03 and NFR-06). |
| Document version | 1.3 (revised after three technical reviews) |
| Status | Draft for team review |

> Items marked **(assumption)** are our own choices, not facts from a standard. They must be stated as assumptions in the report and viva.

---

## 1. Overview

### 1.1 Problem
Most AI room-design tools produce a picture of a room. The picture may look good while the furniture actually overlaps, blocks the door, or leaves no walkable path. Because a picture has no measurable structure, it cannot be checked or scored.

### 1.2 Proposed solution
SpaceGen AI treats a room layout as **structured data** (furniture centre position and rotation, inside a room with a door). A deep generative model proposes many layouts. A gradient-based step then repairs violations, hard rules verify the result, a CNN evaluator ranks the survivors, and the app shows the top 3 with metrics.

### 1.3 What makes this a deep learning project (not just a rules engine)
1. A **Conditional VAE (CVAE)** learns the distribution of good layouts given room size, door and furniture list.
2. A **CNN evaluator** learns to judge layouts from raw top-down images (valid or invalid, quality score).
3. **Latent optimization** back-propagates through the frozen decoder to reduce constraint violations.
4. Every design choice (losses, activations, optimizers, regularizers) is tested in a controlled experiment with plots.

### 1.4 Why a learned model when rules already exist
A rule-based procedural generator can already produce valid layouts, and we build one to create the training data. The learned pipeline is justified by what the generator cannot do well, and we measure each point:
- **Completion around pinned furniture:** the user fixes one item and the rest is arranged around it. A template generator needs a new rejection loop per request; the differentiable decoder plus latent optimization handles it directly (experiment E12).
- **Cost per valid layout:** measured in E1. We do not assume the learned pipeline wins: for requests without a pin, G0 makes the same checker calls without the 150 decoder passes, so it may well be cheaper. If so, we say so.
- **Differentiable repair:** constraint violations are reduced by gradient descent, not just filtered out.

We also state plainly that validity itself is guaranteed by the checker, and that the CNN evaluator is not needed for correctness: it is a controlled CNN study (E9a) and a possible differentiable surrogate (E8b). It is also not a speed-up: reachability for 64 candidates and CNN scoring take about the same time on a laptop, and both are tiny.

---

## 2. Goals and non-goals

### 2.1 Goals
| ID | Goal |
|---|---|
| G1 | Generate multiple diverse furniture layouts for a given room, door and furniture list. |
| G2 | Show measurably that the deep learning pipeline beats the non-neural sampling baselines (B1 uniform random, B2 statistical sampling) on layout validity. The rule-based generator G0 is valid by design and is used only as a reference for cost and diversity. |
| G3 | Provide a working, explainable demo (2D) that a non-expert can use in under a minute. |
| G4 | Justify each deep learning choice mathematically (formula, gradient, behaviour) for the viva. |
| G5 | Deliver a reproducible codebase: fixed seeds, saved configs, one command to regenerate all results. |

### 2.2 Non-goals (explicitly out of scope)
- Photorealistic rendering or image-generation models (Stable-Diffusion style).
- Floor-plan image understanding, multi-room or whole-house generation.
- Natural-language input and text-to-layout (optional stretch only).
- Training on 3D-FRONT or other large real datasets.
- Claims of state-of-the-art results or of novelty over published work.

---

## 3. Users and personas

| Persona | Description | Needs |
|---|---|---|
| **Examiner** (primary) | Course faculty evaluating the project and running the viva | Clear math, reproducible experiments, honest limitations, working demo |
| **Home planner** (demo user) | A student or family member furnishing a room | Enter room size, door and furniture; get a few good layouts fast |
| **Team member** | The two developers | Modular code, clear ownership, easy re-runs |

---

## 4. Scope

### 4.1 Priority levels
- **P0**: must exist for the project to be complete.
- **P1**: should exist; drop only if behind schedule.
- **P2**: optional; attempt only after the Week 2 checkpoint passes.

### 4.2 Scope table
| Priority | Item |
|---|---|
| P0 | Living-room domain: room dimensions, one door, up to 6 furniture slots |
| P0 | Rule-based hard checker (in-room, no overlap, door clearance, reachability, budget) and quality score |
| P0 | Synthetic dataset generator, dataset v1 with train, validation, test and held-out room-size splits |
| P0 | Baselines: B1 uniform random placement, B2 statistical sampling with rejection; G0 procedural generator as an oracle reference (cost per valid layout, diversity reference) |
| P0 | CVAE generator (MLP, multi-head decoder) |
| P0 | CNN evaluator (valid, quality score) |
| P0 | Latent optimization at inference |
| P0 | Top-3 selection with diversity, 2D visualization, Streamlit app |
| P0 | Experiments E1 to E8, E9a and E10 (see Tech Spec), figures, tables, report |
| P1 | E9b (CNN vs feature-MLP); CNN surrogate in latent optimization (M3, E8b) |
| P1 | User-pinned furniture (fix an item, complete the rest). **Highest-priority P1 item**: the strongest argument for the learned approach |
| P1 | Small real-room test set (15 to 20 measured rooms) |
| P2 | Bedroom domain (second room type) |
| P2 | 3D box view (Plotly) |
| P2 | Circulation heatmap visualization |
| P2 | GRU autoregressive decoder as a second generator for comparison |
| P2 | Natural-language input |

---

## 5. User stories and acceptance criteria

| ID | User story | Acceptance criteria |
|---|---|---|
| US-01 | As a home planner, I enter room width, depth, door wall and offset, so the system knows my room. | Inputs are validated (ranges, door inside the wall); invalid inputs show a clear message. |
| US-02 | As a home planner, I choose the furniture I want (and optionally variants and budget). | Selection is checked for budget and footprint feasibility before generation. |
| US-03 | As a home planner, I get 3 different layouts ranked best first. | Three layouts shown, each passing all hard checks, pairwise distinct by at least the diversity threshold. |
| US-04 | As a home planner, I see why a layout is good. | Each layout shows quality score, cost, floor-space use, and pass or fail per hard check. |
| US-05 | As a home planner, I can pin one piece of furniture at a chosen spot (and optionally a facing direction) and let the system arrange the rest. (P1) | The pinned item is snapped exactly onto the requested position, and given the requested facing if one was set, in every output. We report the share of requests for which at least one valid layout is found, split by which item is pinned. |
| US-06 | As an examiner, I see the proposed method compared against baselines on the same input. | A comparison view shows B1, B2, G0, CVAE, and CVAE plus latent optimization with metrics. |
| US-07 | As an examiner, I can reproduce every reported number. | A documented command regenerates data, trains models and rebuilds all tables and figures from fixed seeds. |
| US-08 | As an examiner, I can ask about any formula used. | Report appendix derives every loss, gradient and metric used, verified by unit tests where possible. |
| US-09 | As a home planner, if my room cannot fit the furniture, I am told so. | Infeasible requests return a clear message and a suggestion (remove item, smaller variant), never a bad layout. |

---

## 6. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Accept room type, width, depth, door wall (N, E, S, W), door offset, furniture list, optional variants, optional budget. | P0 |
| FR-02 | Run a feasibility pre-check (total footprint against usable floor area, budget against cheapest selection). | P0 |
| FR-03 | Sample N candidate layouts (default 64) from the CVAE conditioned on the input. | P0 |
| FR-04 | Refine candidates with latent optimization (default 150 Adam steps), togglable in the UI. | P0 |
| FR-05 | Discretize rotation and check all hard constraints (H1 to H5). | P0 |
| FR-06 | Score valid layouts with the CNN evaluator and the rule-based quality score. | P0 |
| FR-07 | Select the top 3 by evaluator score under a diversity constraint. | P0 |
| FR-08 | Render each layout in 2D with labels, door, clearance zone and metrics. | P0 |
| FR-09 | Compare methods (B1, B2, G0, CVAE, CVAE plus latent optimization) on the same input. | P0 |
| FR-10 | Export a layout as JSON and PNG. | P1 |
| FR-11 | Support pinned items (position and optional facing): rotation overwritten before optimization, position snapped exactly before the checks, other items optimized around the pin. | P1 |
| FR-12 | Support the bedroom domain via configuration only (no code change). | P2 |
| FR-13 | Show a 3D box view built from layout JSON. | P2 |

---

## 7. Non-functional requirements

| ID | Requirement | Target (provisional) |
|---|---|---|
| NFR-01 | Inference latency (64 candidates, 150 optimization steps) | at most 5 s on the laptop |
| NFR-02 | CVAE training time (living room) | at most 15 min |
| NFR-03 | CNN evaluator training time | at most 30 min on the GPU (measured by a reviewer: about 16 s per epoch on the RTX 3050 vs about 140 s on CPU at 128x128, batch 256, 42k samples; re-measure) |
| NFR-04 | Reproducibility | Fixed seeds with PyTorch deterministic mode, saved configs, dataset version hash; one seed for screening sweeps and three seeds for headline results; identical results on the same machine, a stated tolerance across machines |
| NFR-05 | Code quality | Modular; unit tests for geometry, rules, rasterizer, KL, overlap gradient |
| NFR-06 | Portability | CVAE, latent optimization, rules and app run on CPU. The evaluator needs the GPU, or a reduced CPU setting (64x64 raster, fewer samples, about 12 epochs in 30 min). Python 3.10 or newer |
| NFR-07 | Explainability | Every model choice has a derivation or experiment in the report |
| NFR-08 | Privacy | No personal data collected; real-room test set contains only room dimensions and furniture sizes |

Targets are provisional and must be checked after the first end-to-end run.

---

## 8. Success metrics

### 8.1 Model and system metrics
| Metric | Definition | Aspiration (calibrate after Week 1 baselines) |
|---|---|---|
| Raw valid rate (RVR) | Fraction of raw samples that pass all hard checks with no filtering. For G0 it is the acceptance rate per attempt. | CVAE beats B1 and B2; CVAE plus latent optimization at least matches CVAE. G0 is valid by design and is not part of this comparison |
| Mean overlap area | Sum of pairwise intersection area per layout (m²) | Lower than every baseline |
| Reachability rate | Fraction of layouts where every item is reachable from the door | Reported for all methods |
| Diversity | Mean pairwise per-slot displacement among valid layouts for one input (m), computed after canonicalization | Reported as a ratio to the diversity of the procedural reference G0; no fixed threshold (independent sampling in B2 is trivially diverse) |
| Cost per valid layout | Time (and attempts) needed to produce one valid layout, for G0, B1, B2, M1, M2 | Reported for all methods. For requests without a pin, G0 is likely cheaper than M2 (M2 makes the same checker calls plus 150 decoder passes); we do not assume M2 wins |
| Mean quality score | Rule-based score (0 to 1) of valid layouts | Higher than baselines |
| Evaluator F1 | Precision, recall and F1 on valid vs invalid (test split) | About 0.90 or better |
| Evaluator ranking agreement | Spearman correlation between evaluator score and rule score | Reported, no threshold |
| Generalization gap | RVR on held-out room sizes minus RVR on in-distribution test | Reported with honest discussion |
| Latency | Seconds per request | At most 5 s |

### 8.2 Project-level success
- All P0 items delivered and reproducible.
- Viva readiness: both members can explain every module and derive every formula in the Tech Spec appendix.
- At least one honest limitation section and one failure-case analysis in the report.

---

## 9. Assumptions and constraints

| # | Assumption or constraint |
|---|---|
| A1 | Team of two, three weeks, roughly equal effort available. |
| A2 | One laptop has the RTX 3050 (VRAM unknown); the other is assumed CPU-only. CPU fallback is required for everything except evaluator training, and GPU time is a shared resource that must be scheduled. |
| A3 | No suitable public dataset with our labels; data is synthetic (rule-generated). |
| A4 | Rooms are axis-aligned rectangles; one door; windows ignored in v1. |
| A5 | Rotations are limited to four directions (0, 90, 180, 270 degrees). |
| A6 | Furniture sizes and prices come from a small illustrative catalog (INR), not real retail data. |
| A7 | Design rules (distance ranges, wall preferences) encode our own assumptions, not an official standard. |
| A8 | Quality-score weights are arbitrary defaults and are configurable. |

---

## 10. Dependencies

- Python packages: PyTorch, NumPy, SciPy, scikit-learn, pandas, Matplotlib, Streamlit, PyYAML, pytest. Plotly for the optional 3D view.
- Papers and repositories used only as ideas (LayoutVAE, constrained layout generation with latent optimization, ATISS, and similar). Code is written by us; papers are cited.
- Access to the college viva and demo format (see open questions).

---

## 11. Risks and mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Scope too large for 3 weeks | High | High | Strict P0/P1/P2 cut lines; Week 2 checkpoint gates all extras |
| R2 | Model only learns our own rules ("just imitating rules") | Certain | Medium | State openly; use several layout styles and jitter; report gains over baselines; test on real rooms |
| R3 | CVAE posterior collapse (all outputs look alike) | Medium | High | KL annealing, small beta values, active-unit diagnostic, beta sweep |
| R4 | Sigmoid saturation near walls hurts positions | Medium | Medium | Experiment E7 (Sigmoid vs Linear plus clamp) |
| R5 | Latent optimization drifts off the data manifold | Medium | Medium | Anchor to each candidate's starting latent `z0`, limited steps, report RVR and diversity together |
| R6 | Evaluator learns shortcuts, F1 looks inflated | Medium | Medium | Held-out room sizes; balanced classes; confusion matrix; per-perturbation-type F1 (E9a), especially near-miss samples |
| R7 | Team member unable to explain the other's code in viva | Medium | High | Cross-teaching sessions; each member owns a derivation set; mock viva |
| R8 | Data generator bugs silently produce invalid "good" layouts | Medium | High | Every generated "good" layout is re-verified by the checker; unit tests |
| R9 | GPU or environment issues | Low | Medium | CPU fallback; pinned requirements file; environment set up on Day 1 |
| R10 | Copying repository code causes licence or viva problems | Low | High | Write own code; cite papers; do not paste repo code |
| R11 | Examiner asks "your generator already makes valid layouts, why train a network?" | High | High | Section 1.4; report cost per valid layout honestly (G0 may win), pinned-furniture completion against G0-pin (E12), and G0 as an explicit reference |
| R12 | Examiner asks what the CNN evaluator adds, since the rule score is exact | High | Medium | State honestly that it is not needed for correctness and not faster than the rule check; justify it with the CNN study (E9a) and the surrogate experiment (E8b) |
| R13 | Effort estimate for P0 is close to total capacity | High | High | P0 needs about 162 h at the midpoint estimates and about 216 h at the top of the ranges, against 168 h available. The reduction ladder in Development_Plan Section 10 (build, report and experiment cuts) closes the 48 h gap exactly, with no slack left, so log actual hours and start it at Gate 1 if you are over; P1 and P2 only if ahead |

---

## 12. Viva and marks alignment (proposal)

The real rubric is not known yet; adjust this table once it is.

| Area | Evidence we will show |
|---|---|
| Problem definition and novelty framing | This PRD, literature notes, honest gap statement |
| Data | Generator design, dataset statistics and histograms, split strategy |
| Model design and justification | Multi-head CVAE with task-to-activation-to-loss table; CNN design; derivations |
| Experiments and analysis | Experiments E1 to E10 (E9b is P1) with tables, curves, gradient plots, failure cases |
| Implementation quality | Repository structure, tests, reproducibility |
| Demo | Streamlit app with method comparison |
| Report and viva | Report with math appendix; both members prepared on all modules |

---

## 13. Deliverables

1. Source repository with README and reproducibility instructions.
2. Dataset generator and dataset v1 (or the seed and config to regenerate it).
3. Trained model checkpoints (CVAE, evaluator).
4. Streamlit demo app.
5. Results: tables (CSV) and figures (PNG) for every experiment.
6. Project report with math appendix, limitations, ethics and deployment notes.
7. Viva preparation sheet (formulas, expected questions, one-page summary).

---

## 14. Open questions for the team

1. What is the exact marks breakdown (report, demo, viva, code)?
2. Is the viva individual or joint, and is a live demo expected?
3. What is the required report format and length?
4. What is the GPU's VRAM (4 GB or 6 GB)? Not blocking, but useful for batch sizes.
5. Is a prior-work comparison expected in the report, or only a literature review?

---

## 15. Glossary

| Term | Meaning |
|---|---|
| CVAE | Conditional Variational Autoencoder; generates data conditioned on inputs such as room size |
| Latent vector z | Compact random code the decoder turns into a layout |
| Latent optimization | Adjusting z by gradient descent so the decoded layout violates fewer constraints |
| Hard checks | Pass or fail rules (in-room, overlap, door clearance, reachability, budget) |
| Quality score | Soft rule-based score in [0, 1] |
| RVR | Raw valid rate: share of unfiltered samples that pass all hard checks |
| Slot | A fixed position in the furniture vector reserved for one furniture type |
