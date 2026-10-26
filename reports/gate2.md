# Gate 2 review (end of Week 2)

| Field | Value |
|---|---|
| Date | 2026-10-06 |
| Tasks covered | T19 to T23, T25 to T30 and T33 (Development Plan Section 4); the P1 tasks T24 and T31 and the P2 task T32 were not started |
| Code | `main` on GitHub; every task commit is listed by `git log` |
| Dataset | `data/v1/`, unchanged since Gate 1 (hash `ef1535ee101e7e8001c0098ce1c5d2e432686933f9dfbd05a7c20f36a7878f83`) |
| Tests | `python run.py test`: 362 tests pass |
| Specification | Tech Spec v1.5 and Development Plan v1.5 record the decisions taken since Gate 1 (change logs) |

## 1. Checklist (Development Plan Section 7)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | CVAE trains without collapse (active units above 0); early stopping test passes | Done | Default run: 8 of 16 latent units active (KL 4.3 nats), no dead hidden units, checkpoint at epoch 104 of 119. `test_early_stopping_cannot_fire_during_annealing` (`tests/test_train_cvae.py`). Collapse appears only where E3b provokes it (Sigmoid, depth 6, BatchNorm off) |
| 2 | Evaluator F1 reported overall and per perturbation type | Done | `reports/tables/evaluator.csv` and `evaluator_per_type.csv`, `reports/figures/evaluator_confusion.png` (`python run.py evaluator-report`): F1 0.973 (valid) and 0.955 (invalid), ROC-AUC 0.991; near-miss overlap is the hardest type (accuracy 81%) |
| 3 | Latent optimization lowers the constraint loss for at least 90% of candidates, and its effect on RVR and diversity is measured | Done | T23, 100 test rooms x 64: 11% of the candidates already meet the constraints at `z0` and keep it, so their loss cannot fall; 99.6% of the others end lower (88.6% of all candidates). E8 measures raw validity (11% at 0 steps, 64% at 200) and the diversity ratio (0.79 to 0.84) |
| 4 | Pipeline returns 3 layouts end to end | Done | `python run.py generate --width 5 --depth 4 --door W --items ...` writes `reports/demo/top1.json` to `top3`; `tests/test_pipeline.py` |
| 5 | E2 to E7 screening results saved; the activation chosen from E3a (BatchNorm on); the best configuration written to `configs/frozen.yaml` | Done | `reports/tables/e2.csv` to `e8.csv` and their figures; the choice in Section 3, `python run.py gate2` |
| 6 | First passes of E1 and E10 run without errors | Done | `reports/tables/e1.csv`, `e10.csv`, `reports/figures/e1_*.png`, `e10_generalization.png` (`python run.py e1`, `python run.py e10`) |
| 7 | Final 3-seed runs on the frozen configuration launched | Done (T33b, 2026-10-08) | `runs/cvae/frozen/seed-0` to `seed-2`, trained from `configs/frozen.yaml`, are bit-identical to the three MAE runs of Section 3; the headline tables are `reports/tables/e1_final.csv`, `e8_final.csv` and `e10_final.csv` |
| 8 | Actual hours logged for Week 2; above 15% over plan, apply the reduction ladder before any P1 or P2 task | **To do** | Fill in Section 5 |
| 9 | Go or no-go recorded for pinned furniture (T38), real rooms (T39), surrogate (T50), 3D (T40), bedroom (T32) | Recorded 2026-10-08 | T38: go, and built (experiment E12). T39: open, it needs 15 to 20 rooms measured by the team. T50, T40, T32: no-go. Reasons in Section 6 |

Also due at Gate 2: cross-teaching session #2 (Section 4). By the plan's rule, extras start only once all nine boxes are ticked and the remaining days still cover the report and viva preparation with a spare day.

## 2. Key numbers

**Models**

| Model | Size | Result |
|---|---|---|
| CVAE, default configuration | 177,732 parameters | about a minute on the GPU; validation loss 0.571; mean position error 0.58 m with `z = mu` |
| CNN evaluator (E9a) | 585,682 parameters | 30 epochs (19 s each on the GPU), best epoch 26; test accuracy 96.6%; Spearman 0.655 with the rule score |
| Latent optimization (M2) | 150 Adam steps on `z` | about 0.9 s per room of 64 candidates on the CPU |

**First pass of E1** (default configuration, 500 Set A test rooms x 64 raw samples; `reports/tables/e1.csv`)

| Method | Raw valid | Quality (valid) | Quality of the top 3 shown | Diversity ratio | Cost per valid layout |
|---|---|---|---|---|---|
| B1 uniform | 10.1% | 0.29 | 0.30 | 0.98 | 17 ms |
| B2 statistical | 57.8% | 0.55 | 0.62 | 1.02 | 3.5 ms |
| G0 generator (reference) | 74.7% per attempt | 0.88 | 0.89 | 1.00 | 2.1 ms |
| M1 CVAE | 10.6% | 0.57 | 0.66 | 0.79 | 15 ms |
| M2 CVAE + latent optimization | **63.6%** | 0.60 | **0.78** | 0.83 | 22 ms |

- M2 beats B1 and B2 on validity (goal G2). **M1 alone does not beat B2**, although the PRD's metric row expected the CVAE to; latent optimization makes the difference. The report says so.
- The evaluator's ranking lifts M2's top 3 from 0.59 (random order) to 0.78; ranking by the exact rule score would give 0.84.
- G0 costs about a tenth of M2 per valid layout, as Tech Spec Section 7 expected.

**First pass of E10** (500 rooms per set; `reports/tables/e10.csv`): M2's raw valid rate is 63.6% in distribution, 66.8% interpolation, 79.5% unseen combination and 81.2% out of range, about 85% of G0's acceptance on every set. The held-out rooms are larger and easier, so "held-out minus in-distribution" is positive and misleading; the report compares each set with G0. M2's top-3 quality falls only out of range (0.74 against 0.78).

## 3. Configuration freeze

**How the choice was made** (`python run.py gate2`, `experiments/gate2.py`)
- **Shortlist from the screening.** The screening (E2 to E8, one seed each) shortlisted the three changes that raised M1's raw valid rate most: Tanh (E3a), the overlap term and MAE (E2). That M1 check sampled test rooms, so the shortlist saw test data; the choice among the shortlisted settings did not.
- **Three seeds on validation rooms.** Each candidate was trained with seeds 0, 1 and 2 (seed 0 is the screening run). Each model was scored as M1 and as M2 on the same 200 Set A validation rooms, 64 samples each, with the same random draws.
- **The rule, fixed before the runs.** A candidate replaces the default if its mean beats the default's by more than the spread (the larger standard deviation of the two) on M2's raw valid rate or on its top-3 quality, and falls short by no more than the spread on the other. Among several, the larger top-3 gain wins. If two or more qualify, the best two are also tried together.

**Results** (mean ± standard deviation over three seeds; `reports/tables/gate2.csv`, per run `gate2_runs.csv`, figure `reports/figures/gate2_candidates.png`)

| Candidate | Position error (`z = mu`) | M1 raw valid | M2 raw valid | M2 quality (valid) | M2 top-3 quality | M2 diversity | Verdict |
|---|---|---|---|---|---|---|---|
| default (ReLU, MSE) | 0.58 m | 11.4 ± 0.8% | 67.1 ± 0.9% | 0.58 | 0.786 ± 0.009 | 2.00 m | the reference |
| Tanh | 0.62 m | 20.0 ± 1.3% | 64.0 ± 2.7% | 0.58 | 0.724 ± 0.009 | 2.00 m | fails: lower on both metrics |
| overlap term (`lambda_overlap` 1.0) | 0.59 m | 19.1 ± 0.8% | 71.8 ± 0.9% | 0.59 | 0.780 ± 0.012 | 2.07 m | passes on validity (+4.7 points) |
| **MAE** | **0.18 m** | 19.1 ± 2.1% | **72.1 ± 0.7%** | **0.66** | **0.850 ± 0.009** | **2.27 m** | **chosen**: +4.9 points, top 3 +0.064 |
| MAE + overlap term | 0.18 m | 24.0 ± 1.0% | 73.0 ± 0.3% | 0.66 | 0.844 ± 0.010 | 2.27 m | passes: +5.9 points, top 3 +0.059 |

**Frozen: the MAE position loss.** `configs/frozen.yaml` is `configs/default.yaml` with `cvae.position_loss: mae`, and nothing else changed.

- **MAE improves both of the rule's metrics.** M2 gives 5 points more valid samples, and the top 3 the user sees rise from 0.79 to 0.85. The quality of all valid samples rises from 0.58 to 0.66, which narrows the gap to G0's 0.88, and diversity rises by 14%.
- **The combination with the overlap term is no better.** Its differences from MAE are within one standard deviation, and it adds a training term, so the rule's pick is also the simpler model.
- **Tanh, the screening's favourite, is worse where it matters.** Its M1 gain holds in every seed (20% against 11% raw valid), but it lowers M2's validity (64% against 67%) and its top-3 quality (0.72 against 0.79). M1 is a poor guide to the deployed M2: choosing on the screening would have picked the wrong activation. Why is untested. One possible explanation is that saturating Tanh units pass smaller gradients from the positions back to `z` during latent optimization.
- **MAE's gain is more than a change of the effective `beta`.** In the screening, `beta` 0.01 with MSE matched MAE's position error, but not its M1 validity or quality (13% and 0.55 against 17% and 0.62, one seed each).
- **MAE's validation loss (0.96) is on another scale than MSE's (0.57).** Compare the position errors instead.

**Decisions** (Tech Spec Section 11)

| # | Decision | Outcome | Evidence |
|---|---|---|---|
| T1 | Hidden activation | ReLU, unchanged | E3a ties ReLU and Tanh on validation loss; Tanh loses on M2 above |
| T2 | Position loss | **MAE** | The comparison above |
| T3 | Position head | Sigmoid, unchanged | E7: no extra error near walls |
| T4 | `beta_target` | 0.1, unchanged | E5: `beta` 0.01 improves reconstruction, not M1's validity or quality |
| T5 | Latent size | 16, unchanged | E5: 4 and 32 change little |
| T6 | Optimization steps | 150, unchanged | E8 on the default model: 63% at 100 steps, 64% at 200; T33b reruns E8 on the frozen model |
| T7 | Bedroom | Not built (P2) | Section 6 |
| T8 | Circulation term | The spec's definition, kept | E10: G0's quality moves only between 0.86 and 0.89 across the test sets, and every set is compared with G0 |

The rest stays at the defaults: Adam at learning rate 1e-3 (E4), dropout 0.1, and early stopping with patience 15 (E6; 200 fixed epochs gained 0.012 in validation loss for three times the training time).

## 4. Cross-teaching sheet #2

Each member explains the *other's* Week 2 modules and derives two formulas on paper. Member B explains Section 4.1, Member A explains Section 4.2.

### 4.1 Member A's modules (explained by Member B)

**`models/cvae.py` (T19): the conditional VAE**
- Encoder `q(z | x, c)`: `[x; c]` (36 + 25 values) through two hidden layers (256 units, BatchNorm, ReLU, dropout 0.1) to `mu` and `log sigma^2` (16 each). Decoder `p(x | z, c)`: `[z; c]` through two hidden layers to a Sigmoid position head (`u, v` per slot) and four rotation logits per slot.
- Loss: masked position MSE over the items present, plus the rotation cross-entropy over items whose rotation is learned, plus `beta` times the KL term.
- *Derive:* `KL(N(mu, sigma^2) || N(0, 1)) = -1/2 sum (1 + log sigma^2 - mu^2 - sigma^2)`, so `dKL/dmu = mu` and `dKL/d log sigma^2 = (sigma^2 - 1) / 2`.
- *Derive:* the reparameterization `z = mu + sigma * eps` gives `dz/dmu = 1` and `dz/d log sigma^2 = sigma * eps / 2`: the sample stays differentiable.
- *Q: What is posterior collapse and how would you see it?* The encoder ignores `x`, the KL of every latent dimension goes to 0 and samples stop depending on `z`. We count active units (KL above 0.01 nats on validation data): 8 of 16 by default, 0 only in E3b's Sigmoid at depth 6 without BatchNorm.

**`train_cvae.py` (T21)**
- `beta` rises from 0 to 0.1 over 20 epochs. The validation loss always uses `beta_target` and a fixed draw of `z`, so epochs compare.
- Early stopping (patience 15) and checkpoint selection start only after annealing. Each epoch logs per-layer gradient norms, dead units and active units.
- *Q: Why can early stopping not fire during annealing?* While `beta` rises the loss can rise with it; the best checkpoint would be one trained mostly without the KL term.

**`latent_opt.py` (T23): M2**
- Minimize over `z`, decoder frozen: `L_c = 10 sum pen^2 + 10 protrusion + 10 door pen^2 (+ 20 pin) + 0.05 * 1/2 ||z - z0||^2`, with Adam (lr 0.05, 150 steps). Rotations stay those decoded at `z0`; a candidate stops once its constraint terms fall below 1e-4.
- *Derive:* `dL/dz = J^T dL/dp`, with `J = dp/dz` the decoder's Jacobian: the geometric gradient on the positions `p` is carried back into latent space.
- *Q: Why anchor to `z0` instead of shrinking `||z||`?* Each candidate stays near its own sample, which keeps the candidates different from one another and close to what the decoder learned.
- *Q: Why fix the rotations?* The arg-max over rotation logits has no gradient.

**Screening E2, E3a, E3b, E5, E7, E8 (`experiments/screening.py`)**
- E2: MAE's gradient has constant size, MSE's grows with the error; MAE also changes the balance with the KL term (the loss scale acts like a different `beta`).
- E3b: `sigma'(z) <= 0.25`, so through six Sigmoid layers the gradient shrinks by up to `0.25^6`: measured 1e-10 at the encoder input. ReLU units with negative input for every sample get no gradient (29% dead at depth 6).
- *Q: Why does BatchNorm hide both effects?* It re-centres and rescales every layer's pre-activations, so Sigmoid works near its steep middle and few ReLU units stay negative for every sample.

### 4.2 Member B's modules (explained by Member A)

**`models/evaluator.py` and `train_evaluator.py` (T20, T22)**
- Four Conv-BatchNorm-ReLU-MaxPool blocks (16, 32, 64, 64 channels) take the 4 x 128 x 128 raster to 64 x 8 x 8, then a 128-unit layer feeds a validity logit and a score.
- Loss: BCE on the validity plus the MSE between the predicted score and the rule score. The trainer keeps the best validation epoch.
- *Derive:* for `p = sigma(z)`, BCE gives `dL/dz = p - y`.
- *Derive:* the spatial size halves at each pooling: 128, 64, 32, 16, 8.
- *Q: Why is near-miss overlap the hardest type?* An overlap of 0.005 m² along a 0.9 m edge is a strip about a tenth of a pixel wide.

**`evaluator_report.py` (E9a)**
- Precision, recall and F1 per class, F1 per perturbation type (invalid as the positive class), ROC-AUC, confusion matrix, Spearman with the rule score.
- *Q: What does a Spearman of 0.655 mean?* The predicted and true scores rank layouts similarly but far from perfectly; E1 shows what that costs: 0.78 against 0.84 for M2's top 3.

**`pipeline.py` (T25)**
- Request checks, variant choice (largest footprint within the budget and `f_max`), M2 sampling, H1 to H4 as the final authority, evaluator ranking, a diverse top 3 (`tau` 0.3 m, relaxed twice by 25%).
- *Q: What does the user see when nothing passes?* A message with a suggestion (remove an optional item, move the door).

**`experiments/headline.py` (T28, T30), E4 and E6 (T27b)**
- E1 gives every method the same rooms; M1 and M2 share their draws, so their difference is the optimization alone. E10 adds G0 per set as the reference for how hard the rooms are.
- E4: Adam scales each parameter's step by a running estimate of its gradient's size; SGD needs a tuned learning rate (0.01 took 153 epochs to reach the target).
- E6: the generalization gap needs both splits scored in eval mode; logged in train mode, dropout makes the training loss look worse than the validation loss.
- *Q: Why measure cost per valid layout?* Validity alone hides that G0 is about ten times cheaper than M2 for unpinned requests.

## 5. Hours log (Week 2)

Planned effort from the Development Plan (S = 1.5 h, M = 3 h, L = 6 h; joint tasks count for both). If your total is more than 15% above plan (A: 31.1 h, B: 29.3 h), start the reduction ladder (Development Plan Section 10) now.

| Task | Owner | Planned (h) | Actual A (h) | Actual B (h) | Notes |
|---|---|---|---|---|---|
| T19 CVAE | A | 6 | | | |
| T20 CNN evaluator | B | 6 | | | |
| T21 CVAE training | A | 3 | | | |
| T22 Evaluator, E9a | B | 3 | | | |
| T23 Latent optimization | A | 6 | | | |
| T25 Pipeline | B | 6 | | | |
| T26 Screening E2, E3a, E3b | A | 6 | | | |
| T27 E5 | A | 1.5 | | | |
| T27b E4, E6 | B | 3 | | | |
| T28 E1 first pass | B | 3 | | | |
| T29 E7, E8 | A | 3 | | | |
| T30 E10 first pass | B | 3 | | | |
| T33 Gate 2 review and freeze | A + B | 1.5 each | | | |
| **Total** | | **A 27, B 25.5** | | | |

## 6. Before Week 3

**Extras: recommendation, for the team to decide (box 9).** Start any of them only once boxes 7 and 8 are ticked and the hours leave room.

| Task | Effort | Recommendation | Why |
|---|---|---|---|
| T38 pinned furniture, G0-pin, E12 (P1) | L (6 h) | **Go**, first | E1 shows G0 is about ten times cheaper per valid layout for requests without pins, so the case for the learned pipeline rests on pinned completion (Tech Spec 7; viva question 2). `latent_opt.py` already has the pin term and the snapping |
| T50 surrogate M3, E8b (P1) | L (6 h) | No-go for now | M2's quality gap to G0 motivates it, but the evaluator's ranking already lifts what the user sees to 0.78, and the optimizer may exploit the surrogate. Revisit only if T38 finishes early |
| T39 real rooms, E11 (P1) | M (3 h) | Go only if you can measure 15 to 20 rooms this week | It needs your measurements; the code exists (`python run.py generate`) |
| T40 3D view (P2) | M (3 h) | No-go | Adds nothing to the experiments or the viva |
| T32 bedroom (P2) | L (6 h) | No-go | Needs heights, prices and the bed's facing, plus a new generator and dataset |

- Run cross-teaching session #2 (Section 4) and fill in the hours (Section 5).
- T33b, done after this review: the frozen configuration trained with three seeds, and the headline E1, E8 and E10 numbers (mean and standard deviation over the seeds). They are in the report, `reports/report.md`.
- Week 3 also built the app (T34, T36), the figures script (T35), the failure analysis (T37) and pinned furniture (T38).
