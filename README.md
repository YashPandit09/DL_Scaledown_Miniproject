# SpaceGen AI

Constraint-aware furniture layout generation with deep learning (Deep Learning mini project).
A conditional VAE proposes layouts for a room, latent optimization repairs constraint
violations, a rule checker verifies every layout, and a CNN evaluator ranks the survivors.

**Start here:** the report, [reports/report.md](reports/report.md) (as a PDF: `reports/pdf/report.pdf`). To see it work:
`pip install -r requirements.txt`, then `python run.py app`. The final models are committed in
`checkpoints/`, so the app runs from a fresh clone without training.

## Results

500 test rooms, 64 raw samples per room, no filtering. M1 and M2 are the mean ± standard deviation
over three training seeds of the frozen configuration.

| Method | Raw valid | Quality of valid layouts | Quality of the top 3 shown | Cost per valid layout |
|---|---|---|---|---|
| B1 uniform random | 10.1% | 0.29 | 0.30 | 18.4 ms |
| B2 statistical sampler | 57.8% | 0.55 | 0.62 | 4.0 ms |
| G0 rule-based generator (reference) | 74.7% per attempt | 0.88 | 0.89 | 2.4 ms |
| M1 CVAE | 17.3 ± 1.9% | 0.66 | 0.78 | 10.5 ms |
| **M2 CVAE + latent optimization** | **69.8 ± 0.9%** | 0.66 | **0.85** | 21.1 ms |

- **Latent optimization is what makes the learned pipeline work:** it takes the valid share from 17% to 70%.
- **The CNN's ranking lifts what the user sees** from 0.67 (random order) to 0.85.
- **The rule-based generator is still better:** about nine times cheaper per valid layout, with
  higher-scoring layouts, and it stays ahead when the user pins a piece of furniture (47% valid
  against 39%). The report says so plainly; the project's value is the controlled study of the
  deep learning parts.
- **Rooms beyond the training distribution:** M2's valid share rises with room size (70% in distribution; 74%, 84% and 87% on the three
  held-out sets), 90 to 93% of the generator's rate on each. Its top-3 quality falls from 0.85 to 0.77
  only in rooms larger than any in training.
- **What the experiments explain:** MAE beats MSE and Huber because of the loss's scale, not
  outliers; BatchNorm hides vanishing gradients and dying ReLU; the raw model's validity is a
  poor guide to the pipeline's (Tanh doubles one and lowers the other); the CNN judges gross
  violations almost perfectly but boundary cases only 81% of the time; reachability, which is not
  differentiable, causes more than half of the pipeline's remaining failures.

## Documents

| File | Contents |
|---|---|
| [reports/report.md](reports/report.md) | **The report**: problem, data, methods, twelve experiments, demo, failure cases, limitations, references |
| [reports/appendix_math.md](reports/appendix_math.md) | Every loss, gradient and metric derived, with the test or experiment that checks it |
| [reports/failure_cases.md](reports/failure_cases.md) | What goes wrong, how often, and a gallery of twelve cases |
| [reports/viva_prep.md](reports/viva_prep.md) | Prepared answers to 26 viva questions, and the one-page sheets |
| [reports/demo_script.md](reports/demo_script.md) | The 5-minute demo, minute by minute, with backups |
| [reports/regeneration.md](reports/regeneration.md) | What was regenerated from the seed and compared with the saved results, and what the comparison found |
| `reports/pdf/` | The five documents above as PDF files (`python run.py pdf`) |
| [PRD.md](PRD.md) | Goals, scope, user stories, success metrics, risks |
| [Tech_Spec.md](Tech_Spec.md) | Data format, rules, models, experiments, math appendix |
| [Architecture.md](Architecture.md) | Modules, data flow, design decisions |
| [Development_Plan.md](Development_Plan.md) | Tasks T01 to T50, gates, cut list, viva preparation |
| [reports/gate1.md](reports/gate1.md) | Gate 1 review: checklist evidence, key numbers, cross-teaching sheet, hours log |
| [reports/gate2.md](reports/gate2.md) | Gate 2 review: checklist evidence, the configuration freeze, cross-teaching sheet #2, hours log, extras |

## Setup

Python 3.11 or newer (tested on 3.12.5); the pinned NumPy and SciPy need 3.11+.

On the GPU laptop, install the CUDA build of PyTorch first, then the rest:

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

On a CPU-only laptop, `pip install -r requirements.txt` is enough. Then check each machine:

```
python run.py check-env
```

It prints the versions, the GPU and its memory, whether two seeded training runs give identical
weights, and how long one evaluator training epoch takes.

## Tasks

Windows has no `make`, so every step runs through `run.py`. Where `make` exists, `make <task>`
does the same thing.

| Command | What it does |
|---|---|
| `python run.py test` | Run the unit tests (extra arguments go to pytest, for example `python run.py test -k geometry`) |
| `python run.py check-env` | Versions, GPU and memory, determinism check, evaluator training speed |
| `python run.py generator-report` | Generate 2,000 Set A layouts, print acceptance, styles and quality, and plot the share of attempts rejected by room area and item count (`reports/figures/generator_rejection.png`; `--layouts N` and `--seed S` change the run) |
| `python run.py set-b-report` | Generate 2,000 Set B layouts and print the share valid and mean quality per perturbation type |
| `python run.py data` | Build the dataset in `data/v1/` (about 7 minutes): Set A (30k), Set B (60k), splits, held-out sets, diversity reference, f_max calibration, `metadata.json` with a content hash, and the figures `reports/figures/dataset_v1_*.png` (`--figures-only` redraws them) |
| `python run.py baselines` | Evaluate B1, B2 and G0 on 500 test rooms of dataset v1, 64 samples each (about 4 minutes); writes `reports/tables/baselines.csv` |
| `python run.py train-evaluator` | Train the CNN evaluator on Set B on the GPU (one epoch by default; `--epochs N --name RUN`); writes `runs/evaluator/RUN/` with `run.json` (seed, configs, git commit, dataset hash), `log.csv` and `model.pt` |
| `python run.py train-cvae` | Train the CVAE on Set A (about 1.5 minutes on the GPU): KL annealing, early stopping, per-epoch log with gradient norms, dead and active units, and a first M1 check; `--name RUN --set cvae.<field>=<value>` for the experiments, `--config configs/frozen.yaml --seed K` for the final runs |
| `python run.py evaluator-report` | E9a metrics of the evaluator in `runs/evaluator/e9a/` on the Set B test split: `reports/tables/evaluator.csv`, `evaluator_per_type.csv`, `reports/figures/evaluator_confusion.png` |
| `python run.py generate --width 5 --depth 4 --door W --items sofa,tv_unit,coffee_table` | The pipeline for one room (M2 by default; `--budget`, `--no-latent-opt`, `name:variant` items, `--pin sofa:2.5,3.4,S` to keep an item where it is): the top 3 as JSON and PNG in `reports/demo/` |
| `python run.py screen all` | The CVAE screening experiments E2 to E8 (one seed per setting, 48 runs, about 75 minutes on the GPU; finished runs are reused): `reports/tables/<experiment>.csv` and `reports/figures/<experiment>_*.png` |
| `python run.py e1` | E1, first pass: B1, B2, G0, M1 and M2 on the same 500 test rooms x 64 samples (about 15 minutes, samplers on the CPU), plus the quality of the top 3 the pipeline would show; per-room rows are cached in `runs/headline/<cvae run>/` (`--cvae RUN`, `--fresh`): `reports/tables/e1.csv`, `reports/figures/e1_*.png` |
| `python run.py e10` | E10, first pass: M1, M2 and G0 on the E1 rooms and 500 rooms of each held-out set (about 30 minutes): `reports/tables/e10.csv`, `reports/figures/e10_generalization.png` |
| `python run.py e1 --tag final --cvae runs/cvae/frozen/seed-0 runs/cvae/frozen/seed-1 runs/cvae/frozen/seed-2` (likewise `e8` and `e10`; `e10 --sets` splits the run) | The headline runs (T33b) on the three seeds of the frozen configuration: M1 and M2 as the mean and standard deviation over the seeds, B1, B2 and G0 once; `reports/tables/e1_final.csv`, `e8_final.csv`, `e10_final.csv` and their figures. About 2.5 hours in all; keep the laptop on mains power, because E1's costs are timings |
| `python run.py gate2` | Gate 2 freeze: three seeds of the shortlisted CVAE settings (trains only missing runs), M1 and M2 scored on 200 Set A validation rooms, the rule from `reports/gate2.md` applied; writes `configs/frozen.yaml`, `reports/tables/gate2*.csv`, `reports/figures/gate2_candidates.png` (about an hour) |
| `python run.py e12` | E12, pinned furniture (T38): 60 requests per item, each with one item pinned where a generator layout of the room has it; M2, M1, G0-pin, B1 and B2, timed in turn: `reports/tables/e12.csv`, `reports/figures/e12_pinned.png` (about 35 minutes) |
| `python run.py failures` | Failure-case analysis (T37): B1, B2, M1 and M2 sample 100 test rooms; every sample is classified by the hard check it breaks, or as valid but poor; `reports/tables/failure_causes.csv`, `failure_cases.csv`, `reports/figures/failure_cases.png`, discussed in `reports/failure_cases.md` (about 3 minutes) |
| `python run.py figures` | Redraw every figure from the saved tables and logs in seconds, without training or sampling (`--tables` first rebuilds the screening tables from the saved runs). The per-epoch logs of the screening runs are copied to `reports/logs/screen/`, so the training curves can be redrawn from a fresh clone |
| `python run.py demo-assets` | The demo's backup pictures and the report's demo figures, made by the app's own code: `reports/demo/layouts.png`, `pinned.png`, `compare.png` |
| `python run.py all` | Regenerate every result in order, about eight hours (`--list` prints the steps; `--from STEP --to STEP` runs a part). Set `SPACEGEN_OUTPUT` to an empty folder first, so it starts from nothing and leaves the saved results alone |
| `python run.py check-regeneration` | Compare regenerated tables (outside their timing columns), model hashes, the dataset hash and the committed checkpoints with the saved ones (`--record` saves the hashes) |
| `python run.py export-checkpoints` | Copy the final models from `runs/` to `checkpoints/` |
| `python run.py pdf` | The report, the math appendix, the failure cases, the viva notes and the demo script as A4 PDF files in `reports/pdf/`, with typeset formulas, captions and page numbers. It prints through a headless Edge or Chrome and needs an internet connection for the formulas (MathJax). Run it again after changing a document |
| `python run.py app` | The Streamlit demo (T34, T36): the room, door, furniture, budget and options in the sidebar; the top 3 layouts with their rule quality and evaluator score, cost, floor use, each hard check and JSON and PNG exports; the five methods compared on the same room; the saved figures. It opens without trained models and says what to run |
| `python -m spacegen.viz layout.json layout.png` | Draw a layout JSON as a floor plan with its hard-check results |

## Reproducibility

- `spacegen.seed.set_seed(seed)` seeds Python, NumPy and PyTorch and turns on PyTorch's
  deterministic mode, so two runs with the same seed on the same machine give bit-identical
  weights (tested for CPU and CUDA in `tests/test_seed.py`).
- Import `spacegen` before doing any CUDA work. Importing it sets `CUBLAS_WORKSPACE_CONFIG`,
  which only takes effect if it is set before the first GPU matrix operation; `set_seed`
  raises an error if CUDA was used first.
- Generated data (`data/`) and training runs (`runs/`) are not committed; they are rebuilt
  from the seed and the configs in `configs/`.
  `data/v1/metadata.json` records the seed, the configs, the git commit and a SHA-256 hash of
  the data contents; a rebuild on the same machine gives the same hash.
- The final models (the three frozen CVAE seeds and the evaluator, 4.7 MB) are committed in
  `checkpoints/`, and `reports/model_hashes.json` holds the hash of every trained model.
- `python run.py all` regenerates everything and `python run.py check-regeneration` compares the
  result with what is saved. Checked on 9 October 2026: a fresh clone from GitHub passes
  every test without `data/` or `runs/`, generates from the committed checkpoints, redraws the figures
  byte for byte and rebuilds dataset v1 with the identical hash. A complete regeneration into an empty folder
  then reproduced all 21 tables outside their timing columns, all 64 trained models, the dataset and the
  Gate 2 choice exactly (`reports/regeneration.md`).
- Timings (cost per valid layout, seconds per room) are measured with the methods taking turns
  on the same rooms, because the laptop's speed changes with its power and thermal state. Keep
  it on mains power for long runs.

## Status

- [x] T01 Repository, seed helper, machine check
- [x] T02 Geometry
- [x] T03 Catalog (living room; the bedroom is P2)
- [x] T04 Rules file: every threshold in `configs/rules.yaml`, each marked as an assumption
- [x] T05 Door geometry and hard checks H1 to H3
- [x] T06 2D floor plans and the layout JSON format
- [x] T07 Evaluator input: 4-channel fractional-coverage rasters, built on the GPU
- [x] T08 Reachability (H4)
- [x] T09 Quality score (alignment, relations, circulation, space)
- [x] T10 Layout vectors (condition c: 25 values, target x: 36), canonical form, dataset tensors on the GPU
- [x] T11 Set A generator with all three living-room styles: (a) wall sofa, (b) floating sofa,
  (c) L-shape (styles (b) and (c) moved here from T13, which keeps the Set B perturbations);
  furniture that grows with room area; rejection report
- [x] T13 Set B for the evaluator: half generator layouts, half perturbed (jitter, rotation, random,
  forced overlap, near-miss at the overlap tolerance and at the door zone), each labelled by the checker
- [x] T14 Splits 70/15/15; held-out rooms (interpolation 22 to 26 m², above 32 m², out of range) that
  training never sees; diversity reference of generator layouts
- [x] T12 Baselines: B1 uniform random, B2 statistical sampler fitted on the Set A training split,
  G0 the generator
- [x] T15 Dataset v1: 30k Set A, 60k Set B (62% valid), held-out sets, metadata and content hash,
  figures; f_max calibrated to 0.38 (the spec's start was 0.45)
- [x] T16 Evaluation harness (`spacegen/metrics.py`, `spacegen/evaluate.py`) and the baseline table:
  raw valid B1 10%, B2 58%, G0 75% per attempt
- [x] T17 Notebook `notebooks/01_manual_backprop.ipynb`: hand-derived gradients of a two-item toy
  network (masked MSE, cross-entropy, penetration-depth overlap) equal autograd to 2e-15
- [ ] T18 Gate 1 review: 7 of 9 boxes done with evidence in `reports/gate1.md`; the cross-teaching
  session and the Week 1 hours log are for the team
- [x] T19 CVAE (`spacegen/models/cvae.py`): encoder, decoder with a logit rotation head, masked loss
  and KL, every Week 2 experiment switch in `configs/default.yaml`; 177,732 parameters; memorizes 64
  layouts within 500 steps
- [x] T20 CNN evaluator (`spacegen/models/evaluator.py`, `spacegen/train_evaluator.py`): 585,682
  parameters, Set B rasterized on the GPU, 19 s per epoch; validation accuracy 94% after 10 epochs
- [x] T21 CVAE training (`spacegen/train_cvae.py`): default run stops at epoch 119 (checkpoint 104), 8 of 16
  latent units active, no dead units; first M1 check 12% raw valid (before tuning and latent optimization)
- [x] T22 Evaluator, E9a (30 epochs, best epoch 26): test accuracy 96.6%, F1 0.973, ROC-AUC 0.991; hardest
  type near-miss overlap (accuracy 81%); Spearman with the rule score 0.66
- [x] T23 Latent optimization (`spacegen/latent_opt.py`, M2): on 100 test rooms x 64 samples, raw valid
  12% (M1) to 65% (M2); 99.6% of the candidates that needed repair end with a lower loss
- [x] T25 Pipeline (`spacegen/pipeline.py`, `python run.py generate`): request checks, variants within
  the budget and f_max, M2 sampling, checks, evaluator ranking, diverse top 3; about 1.5 s per request
- [x] T26 Screening E2, E3a, E3b (30 runs, one seed each; `experiments/`): MAE has by far the lowest
  position error (0.19 m vs 0.58 MSE), partly because the loss scale shifts the KL balance; E3a ties
  ReLU and Tanh on validation loss while Tanh gives M1 22% raw valid vs 12%; E3b shows vanishing
  gradients (Sigmoid depth 6: 1e-10 at the input) and dying ReLU (29%) without BatchNorm, both masked by it
- [x] T27, T27b, T29 Screening E4 to E8: Adam 1e-3 converges fastest to the best loss; beta 0.01 cuts the
  position error from 0.58 m to 0.18 m and raises diversity; the CVAE does not overfit (gaps below 0.01);
  the Sigmoid head does not suffer near walls; latent optimization reaches 53% raw valid at 25 steps and
  64% at 200, without losing diversity
- [x] T28 E1, first pass (default configuration, 500 test rooms x 64 samples; `experiments/headline.py`): raw valid
  B1 10%, B2 58%, M1 11%, M2 64% (G0 75% per attempt). M2's valid layouts score 0.60 on quality (G0 0.88), but the
  top 3 the evaluator picks score 0.78 (0.84 if ranked by the rule score, 0.59 in random order). M2 costs 22 ms
  per valid layout, G0 2 ms
- [x] T30 E10, first pass: M2's raw valid rate rises with room size (64% in distribution, 67% interpolation, 79%
  unseen combination, 81% out of range) as G0's does (75% to 96%), so the held-out rooms are easier, not harder;
  M2's quality holds (0.60 to 0.56; the top 3 from 0.78 to 0.74)
- [ ] T33 Gate 2 review (`reports/gate2.md`): 8 of 9 boxes done. Configuration frozen (`configs/frozen.yaml`, MAE
  position loss) after comparing three seeds of the screening's shortlist with M2 on 200 validation rooms: MAE gives
  72% raw valid against 67% and a top-3 quality of 0.85 against 0.79; Tanh nearly doubles M1's validity but lowers
  M2's. Left for the team: the Week 2 hours
- [x] T33b Final runs on the three frozen seeds (bit-identical to the Gate 2 MAE runs): E1, E8 and E10 with M1 and
  M2 as mean and standard deviation; timings from a pass in which the methods take turns. M2 69.8 ± 0.9% raw valid
  against B2's 57.8%; G0 about nine times cheaper per valid layout
- [x] T34, T36 Streamlit app (`python run.py app`, `app/streamlit_app.py`, logic in `spacegen/app_logic.py`): room, door,
  furniture, budget and an optional pinned item in the sidebar; the top 3 with quality, cost, floor use, each hard
  check and JSON and PNG exports; the five methods on the same room; the saved figures. `tests/test_app.py` runs
  the app headless, also without trained models
- [x] T35 `python run.py figures` redraws every figure from the saved tables and logs in seconds; the screening
  runs' per-epoch logs are copied to `reports/logs/screen/`, so a fresh clone can redraw the training curves
- [x] T37 Failure analysis (`python run.py failures`, `reports/failure_cases.md`): latent optimization removes 87% of
  the out-of-room failures and 89% of the overlaps, but only 37% of the unreachable items, which it does not target;
  18.5% of M2's samples fail on reachability alone
- [x] T38 Pinned furniture (P1, the one extra taken): pins in the pipeline, the baselines and the app; E12 on 360
  requests. The pin always holds and 99% of requests get a valid layout, but the generator with the item moved
  (G0-pin) beats M2 on validity (47% against 39%), quality and cost: the expected case for the learned pipeline
  is not supported
- [x] T41 Math appendix (`reports/appendix_math.md`)
- [x] T42a, T42b, T43 Report (`reports/report.md`): ten sections with the limitations, ethics and references
- [x] T44 Fresh-clone check: a clone from GitHub passes every test, generates from the committed checkpoints, redraws
  every figure byte for byte and rebuilds dataset v1 with the identical hash; the frozen seeds retrain bit-identically
  (`reports/regeneration.md`)
- [x] T46 Full regeneration and the `final` tag: every step, run again into an empty folder with nothing reused,
  reproduced all 21 tables outside their timing columns, all 64 trained models, the dataset and the Gate 2 choice
  exactly, and 24 of 28 figures byte for byte (the other four plot measured times). `reports/regeneration.md` is the
  log, with the three small defects the run exposed in our own tooling. The repository is tagged `final`
- [x] `python run.py pdf` builds the report, the math appendix, the failure cases, the viva notes and the demo script
  as PDF files (`reports/pdf/`)
- [ ] T45, T47 Viva answers and cheat sheets (`reports/viva_prep.md`) and the demo script (`reports/demo_script.md`)
  are written; learning them, the rehearsals and the mock viva are for the team
- [x] Screenshots of the app at each step of the demo (`reports/demo/app_*.png`), as the backup for the demo and
  for the report's Section 6. Looking at them showed a cost figure cut off and plan labels too small to read; both
  are fixed, and the app is light with the figures' blue as its accent (`.streamlit/config.toml`)
- [ ] T48 Submission: for the team. Put your names in line 3 of `reports/report.md` and run `python run.py pdf`
  again. Still open besides: the hours logs and cross-teaching sessions, and T39 (real rooms), which needs rooms
  measured by the team
- Not built: T24 and T31 (feature MLP), T50 (surrogate), T40 (3D view), T32 (bedroom)
