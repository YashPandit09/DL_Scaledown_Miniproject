# Gate 1 review (end of Week 1)

| Field | Value |
|---|---|
| Date | 2026-10-02 |
| Tasks covered | T01 to T17 (Development Plan Section 3) |
| Code | `main` on GitHub; every task commit is listed by `git log` |
| Dataset | `data/v1/`, hash `ef1535ee101e7e8001c0098ce1c5d2e432686933f9dfbd05a7c20f36a7878f83`, built at commit `200ec64` with a clean tree |
| Tests | `python run.py test`: 278 tests pass |
| Specification | Tech Spec and Development Plan v1.4 record every decision taken while building (change logs) |

## 1. Checklist (Development Plan Section 7)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Geometry (including the containment gradient), rules, door placement and rasterizer tests pass | Done | `tests/test_geometry.py` (32 tests, among them `test_contained_item_gets_a_push_that_the_exact_area_cannot_give`), `tests/test_rules.py` (48, among them `test_door_stays_on_its_wall_away_from_the_corners` for all four walls), `tests/test_raster.py` (15: exact and 32x supersampled references, GPU equals CPU, gradient check) |
| 2 | Reachability passes on an empty room and the start cell is passable | Done | `test_empty_room_is_walkable_from_any_door` (every wall, door at both extreme offsets; asserts the start cell is passable); 19 reachability tests in all |
| 3 | Every Set A layout passes the checker and is canonical (asserted) | Done | `test_every_layout_passes_the_hard_checks_and_is_canonical` (`tests/test_generator.py`) and `test_saved_sets_are_canonical_and_leak_free` (`tests/test_build_dataset.py`); the generator keeps only checked layouts |
| 4 | Dataset v1 generated, hashed, split, with separately sampled held-out sets | Done | `python run.py data` (about 7 minutes). A rebuild from the committed code gave the identical hash. `test_training_data_never_contains_a_held_out_room` asserts there are no leaks |
| 5 | Histograms and the rejection-rate plot saved; `f_max` calibrated | Done | `reports/figures/dataset_v1_rooms.png`, `dataset_v1_set_b.png`, `dataset_v1_rejection.png`, `dataset_v1_f_max.png`; `f_max = 0.38` in `configs/rules.yaml` |
| 6 | B1, B2 and G0 baseline numbers saved, including cost per valid layout | Done | `reports/tables/baselines.csv` (`python run.py baselines`) |
| 7 | Manual backprop notebook matches autograd | Done | `notebooks/01_manual_backprop.ipynb`: largest difference 1.8e-15 against autograd, 1.8e-9 (relative) against finite differences; `tests/test_notebooks.py` runs it |
| 8 | Each member can explain the other's Week 1 modules | **To do** | Cross-teaching session with Section 4 |
| 9 | Actual hours logged for Week 1; above 15% over plan, start the reduction ladder | **To do** | Fill in Section 5 |

## 2. Key numbers

**Dataset v1**

| Part | Size | Notes |
|---|---|---|
| Set A (CVAE) | 30,000 layouts from 30,206 rooms | 57% of placement attempts kept; 206 rooms (0.7%) dropped after 20 failed attempts; styles: wall sofa 46%, floating sofa 48%, L-shape 6%; mean quality 0.88 |
| Set B (evaluator) | 60,000 layouts, 62% valid | clean 100%; jitter 0.1 / 0.3 / 0.6 m: 47 / 31 / 23%; rotation 29%; random 12%; forced overlap 0%; near-miss overlap 49%, near-miss door 50% |
| Splits | 70 / 15 / 15 of each set | 21,000 / 4,500 / 4,500 for Set A; 42,000 / 9,000 / 9,000 for Set B |
| Held-out sets | 1,000 rooms each | interpolation (22 to 26 m²), unseen combinations (above 32 m²), out of range (W 7 to 8 m, D 6 to 7 m) |
| Diversity reference | 3,992 layouts | 20 generator layouts for each of 200 Set A test rooms (a few runs found no layout) |
| `f_max` | 0.38 | Calibrated 0.377 on 3,903 rooms; the spec's starting value was 0.45 |

Above 32 m² almost every layout uses the floating sofa (99% in the unseen-combination set, 100% out of range), because a wall sofa needs one side of the room to be at most 4.8 m. E10 on those sets therefore mostly tests one style.

**Baselines** (500 Set A test rooms, 64 raw samples per room, same rooms for every method)

| Method | Raw valid | Overlap per sample | Reachability | Quality (valid) | Valid per room | Diversity ratio | Cost per valid layout |
|---|---|---|---|---|---|---|---|
| B1 uniform | 10.1% | 0.21 m² | 0.55 | 0.29 | 6.4 | 0.98 | 9.9 attempts, 20 ms |
| B2 statistical | 57.8% | 0.002 m² | 0.70 | 0.55 | 37.0 | 1.02 | 1.7 attempts, 4 ms |
| G0 generator | 74.7% per attempt | n/a | n/a | 0.88 | 47.8 | 1.00 | 1.3 attempts, 2 ms |

B2 is often valid but its layouts score low on quality: it avoids collisions without getting the arrangement right. G0's diversity ratio of 1.00 against its own reference confirms the metric. G0's outputs are valid by construction, so overlap and reachability are not reported for it.

**Hardware** (measured in T01): the RTX 3050 6 GB trains an evaluator-sized CNN at about 1,960 samples/s in deterministic mode (21 s per 42k-sample epoch, 1.4 GiB peak); the CPU manages 100 to 300 samples/s (140 to 430 s per epoch). Train the evaluator on the GPU.

## 3. Decisions taken while building

The Tech Spec v1.4 change log has the details; these matter most for the report and the viva.

- **Reachability checks the whole front face (access line), not one point.** The single point in front of the face's centre failed every sofa with a coffee table in front of it.
- **All three generator styles are in T11.** With the 1.5 to 3.5 m sofa-TV range, a wall sofa fits only rooms with one side of 2.9 to 4.8 m, about three quarters of rooms. T13 kept only the Set B perturbations.
- **Optional furniture grows with room size**: each optional item appears with probability 0.25 in the smallest room, rising to 0.75 at 30 m² and above, so every combination still occurs at every size.
- **Overlap near-misses are drawn from 0.002 to 0.0125 m²** (log-uniform), so half fall under the 0.005 m² tolerance. The spec's 0.002 to 0.05 m² would make at least 70% of them invalid.
- **`f_max` is 0.38**: the footprint ratio at which the generator can still furnish half of the rooms.
- **B1 keeps every item inside the room**, so it does not fail on the walls alone.
- **B2 redraws items that stick out of the room**, chosen after T15. Without it B2 is 20% valid, mostly because positions taken as fractions of a larger training room put wall items through a smaller room's wall. With it, 58%.
- **Open:** the circulation score cannot reach 1 even in an empty room (0.68 to 0.83, depending on room size), which favours large rooms in cross-room comparisons. The spec's definition is kept for now; decide before E10 (Tech Spec Section 11).

## 4. Cross-teaching sheet

The Development Plan asks each member to explain the *other's* modules and derive two formulas on paper. Member B explains Section 4.1, Member A explains Section 4.2. For each module: what it does, the decisions behind it, and questions to practise.

### 4.1 Member A's modules (explained by Member B)

**`geometry.py` (T02): footprints, overlap, penetration depth**
- A footprint is a centre plus an effective size; rotation classes 1 and 3 swap width and depth.
- The checker uses the exact overlap area. The losses use the penetration depth `pen = min(px, py)`, `px = ReLU((w_i + w_j)/2 + mu - sqrt((x_i - x_j)^2 + 1e-6))`.
- One implementation serves NumPy (checker, generator) and PyTorch (losses).
- *Derive:* `d px / d x_i = -(x_i - x_j) / sqrt((x_i - x_j)^2 + 1e-6)` where `px > 0`; the exact area's gradient is zero when one box lies inside the other along x.
- *Q: Why not train on the overlap area?* A contained item gets no gradient (Appendix K). Show `test_contained_item_gets_a_push_that_the_exact_area_cannot_give` or the notebook's last cell.

**`raster.py` (T07): the evaluator's input**
- Four channels (room, furniture summed, item fronts, door zone) on a fixed 8 m canvas at 128 px, so a metre is always 16 pixels.
- Each pixel holds the exact share of its area a box covers: the outer product of the box's overlap with the pixel's column and row. That is one batched matrix product on the GPU, and it is differentiable in the positions.
- The front band is 0.10 m thick and drawn on every equivalent face of a symmetric item.
- *Q: Why fractional coverage?* A 1 cm move must change the input, and overlaps must show as values above 1.

**`dataset.py` (T10): vectors and canonical form**
- Condition `c` (25): W/8, D/8, door wall one-hot (N, E, S, W), door offset, six presence masks, six widths/3, six depths/3.
- Target `x` (36): per slot `(u, v)` with `u = x/W`, `v = y/D`, then the one-hot rotation.
- Canonical form: a symmetric item's rotation is reduced modulo `4 / rot_symmetry`; absent slots are zeros.
- The whole dataset sits on the GPU and mini-batches are drawn by index.
- *Q: What goes wrong without canonical labels?* The same layout gets several targets (a coffee table at 0 or 180 degrees), so the network is trained towards an average of contradictory answers.

**`splits.py` (T14): splits and held-out sets**
- Training rooms are drawn again whenever they fall in a held-out region. Each held-out set is drawn inside its own region. Set A and Set B are split 70/15/15.
- *Q: Unseen combination vs out of range?* Above 32 m², each width and depth also occurs in training, just not together. Out of range, W and D themselves are beyond the training ranges.

**`baselines.py` (T12), `metrics.py` and `evaluate.py` (T16)**
- B1: uniform placement inside the room. B2: per-item `(u, v, rotation)` drawn from similar training rooms, with redraws on overlap or protrusion. G0: the generator.
- Every method gets the same rooms and 64 raw samples per room. G0 runs single attempts, so its valid rate is acceptance per attempt.
- Diversity: the mean over pairs of the mean over shared slots of (centre distance + 0.5 m if the rotation differs).
- *Q: Why is G0 left out of the validity comparison?* It filters through the checker internally, so it is valid by construction. It is the reference for cost and diversity.

**`notebooks/01_manual_backprop.ipynb` (T17)**
- *Derive:* the Sigmoid head `delta = g * p * (1 - p)`; softmax with cross-entropy `q - onehot(r)`; a linear layer `dL/dW = h^T delta`, `dL/dh = delta W^T`; ReLU `delta * [z > 0]`; the mask multiplies everything an absent item contributes by zero.

### 4.2 Member B's modules (explained by Member A)

**`catalog.py` (T03) and `configs/rules.yaml` (T04)**
- Six living-room slots, variants with sizes and prices, and flags (`rot_symmetry`, `needs_access`, interchangeable groups). The loader rejects mistakes: a 4-way symmetric item must be square, so turning it never changes its footprint.
- Every threshold in `rules.yaml` is our own assumption and is marked as such (PRD A7).
- *Q: Why is the TV unit `needs_access: false`?* It is viewed from the sofa, not walked up to; requiring access would reject valid wall layouts.

**`rules.py` (T05, T08): door and hard checks H1 to H4**
- Door centre `s = m_d + (L - 2 m_d) o` with `m_d = 0.65 m`, so the door never touches a corner. The clearance zone is 0.9 x 0.9 m in front of it.
- H1 in room (1 mm tolerance), H2 pairwise overlap at most 0.005 m², H3 door zone clear.
- H4: on a 0.10 m grid a cell is walkable if it is at least 0.3 m from every item and wall; the walkable cells connected to the door zone's centre form the reached region. An item is reachable if a reached cell lies within 0.3 m of its access line (the front face moved 0.35 m out), on any equivalent face.
- *Derive:* the door centre at `o = 0` and `o = 1` is 0.65 m from the corners.
- *Q: Why start at the centre of the door zone?* The cell just inside the door is closer than 0.3 m to the wall, so it is never walkable.

**`quality.py` (T09): the score S**
- `S = 0.30 align + 0.30 relations + 0.25 circulation + 0.15 space`.
- Four relations (sofa-TV distance, coffee table in front of the sofa, side table beside it, armchair at 90 degrees facing the table). Each scores 1 inside its range and falls linearly to 0 at 0.5 m outside it.
- Space scores 1 for 15 to 40% floor coverage.
- *Q: Why does alignment ignore the orientation of a side table?* Its stored rotation is arbitrary (canonical form), so only its position can be judged.

**`generator.py` (T11): Set A and G0**
- Each attempt picks a style that fits the room. Items are placed in the frame of the wall behind the sofa. Every item placed relative to the sofa limits where the sofa can stand, and its position is drawn from what is left. The bookshelf goes last, on a free stretch of wall.
- The room stays fixed for up to 20 attempts, so the rooms in the data follow the designed distribution. Only 0.7% are dropped.
- *Q: Why can a wall sofa not serve a 7 x 6 m room?* Sofa and TV unit against opposite walls would be at least 4.7 m apart front to front, beyond the 3.5 m range.

**`perturb.py` (T13): Set B**
- Half clean, half perturbed: jitter, rotation, random, forced overlap, near-miss.
- A near-miss slides one item until the overlap (or the gap to the door zone) sits right at the checker's boundary, touching nothing else and keeping every item reachable.
- *Q: Why near-misses?* An overall F1 hides the hard cases; the per-type F1 on near-misses shows what the CNN can see at the raster's resolution (a 0.005 m² overlap along a 0.9 m edge is a strip about a tenth of a pixel wide).

**`build_dataset.py` (T15): dataset v1 and `f_max`**
- Each part draws from its own random stream spawned from the seed. The hash covers the arrays inside the `.npz` files, not the files, whose zip timestamps change.
- `f_max`: a logistic fit of "the generator furnished the room within 20 attempts" against the footprint ratio crosses one half at 0.377.
- *Q: Why spread the calibration rooms evenly over the ratio?* Random rooms are almost never crowded (35 of 3,000 above a ratio of 0.3), so the fit would have no data where it matters.

## 5. Hours log (Week 1)

Planned effort from the Development Plan (S = 1.5 h, M = 3 h, L = 6 h; joint tasks count for both). Log the time each of you actually spent, including reviewing and learning code you did not write. If your total is more than 15% above plan (A: 29.3 h, B: 31.1 h), start the reduction ladder (Development Plan Section 10) now.

| Task | Owner | Planned (h) | Actual A (h) | Actual B (h) | Notes |
|---|---|---|---|---|---|
| T01 Repository, seed helper, machine check | A + B | 1.5 each | | | |
| T02 Geometry | A | 3 | | | |
| T03 Catalog | B | 1.5 | | | |
| T04 Rules file | B | 1.5 | | | |
| T05 Door and hard checks H1 to H3 | B | 3 | | | |
| T06 Floor-plan plots | A | 3 | | | |
| T07 Rasterizer | A | 3 | | | |
| T08 Reachability H4 | B | 3 | | | |
| T09 Quality score | B | 3 | | | |
| T10 Layout vectors | A | 3 | | | |
| T11 Generator (all three styles) | B | 6 | | | |
| T12 Baselines | A | 3 | | | |
| T13 Set B | B | 3 | | | |
| T14 Splits | A | 1.5 | | | |
| T15 Dataset v1 | B | 3 | | | |
| T16 Baseline evaluation | A | 3 | | | |
| T17 Backprop notebook | A | 3 | | | |
| T18 Gate 1 review | A + B | 1.5 each | | | |
| **Total** | | **A 25.5, B 27** | | | |

## 6. Before Week 2

- Run the cross-teaching session (Section 4) and fill in the hours (Section 5); then tick items 8 and 9 in Development Plan Section 7.
- Decide the circulation score before E10 (Tech Spec Section 11, decision T8).
- Week 2 starts with T19 (CVAE) and T20 (CNN evaluator). The evaluator trains on Set B on the GPU; the CVAE trains on Set A and also runs on a CPU.
