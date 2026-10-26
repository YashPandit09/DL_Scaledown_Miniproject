# SpaceGen AI: Technical Specification

| Field | Value |
|---|---|
| Version | 1.6 (the project as built: decisions taken through Week 3; see change log in Section 12) |
| Companion docs | PRD.md, Architecture.md, Development_Plan.md |
| Scope | Living room (P0). Bedroom (P2). Same code, different config. |

> All numeric values marked **(start)** are starting points to be tuned. Values marked **(assumption)** are our own design choices and must be presented as such.

---

## 1. Conventions

- **Units:** meters internally. Angles are discrete (see rotation).
- **Room:** axis-aligned rectangle of width `W` (x-axis) and depth `D` (y-axis), origin at bottom-left, `x in [0, W]`, `y in [0, D]`.
- **Door:** on one wall (N, E, S, W), fixed door width `0.9 m` **(assumption)**. Walls run counter-clockwise and the *start corner* of each wall is: S wall `(0,0)` to `(W,0)`; E wall `(W,0)` to `(W,D)`; N wall `(W,D)` to `(0,D)`; W wall `(0,D)` to `(0,0)`. The door **centre** lies at distance `s = m_d + (L - 2*m_d) * o` from the start corner, where `L` is the wall length, `o in [0,1]` is the door offset and `m_d = 0.65 m` (half the door width plus a 0.2 m margin). The door therefore never touches a corner: its centre is at least 0.65 m and its edges at least 0.2 m from any corner.
- **Door clearance zone:** rectangle of door width by `0.9 m` **(assumption)** projecting into the room from the door. Furniture must not enter it.
- **Rotation class `r in {0,1,2,3}`:** the direction the item's front faces: 0 = North (+y), 1 = East (+x), 2 = South (-y), 3 = West (-x). Classes 1 and 3 swap the item's width and depth in the footprint.
- **Effective footprint:** `(w_eff, d_eff) = (w, d)` if `r` is even, else `(d, w)`.
- **Canonical labels:** some items look identical after certain rotations or swaps, which would give the network contradictory targets. The catalog stores `rot_symmetry` per item (1 = none, 2 = same after 180 degrees, 4 = same after any 90 degrees) and an optional interchangeable `group` of slots. The generator always writes the canonical form: `r` is reduced modulo `4 / rot_symmetry` (a coffee table only uses classes 0 and 1, a side table always 0), and items in an interchangeable group (the bedroom nightstands) are ordered along the bed's lateral axis (perpendicular to the direction the bed faces), so the order does not depend on jitter when the bed is against an east or west wall. The rotation loss is masked for `rot_symmetry = 4` items, and diversity ignores non-informative flips (a requested facing for a symmetric pinned item is compared modulo its symmetry). Because the stored rotation is arbitrary among equivalent faces, the checker, the raster and the quality score treat **all equivalent faces of a symmetric item alike** (Sections 3.3, 3.4 and 4.2). Absent slots are zero-filled in the canonical form. Interchangeable groups are canonicalized once the bedroom is built (P2); until then the code refuses a catalog with groups rather than produce unordered labels.
- **Normalized coordinates:** `u = x / W`, `v = y / D`. The model predicts `(u, v)`; all geometry (overlap, rules) is computed in meters after `x = u*W`, `y = v*D`.

---

## 2. Data specification

### 2.1 Catalog (`configs/catalog.yaml`)
Each item: `id`, `slot`, `name`, `w`, `d`, `h` (m), `price` (INR), optional `variant`. Values below are **illustrative placeholders (assumption)**.

**Living room (K = 6 slots)**

| Slot | Item | Variants (w x d x h, m; price INR) |
|---|---|---|
| 0 | sofa | 3-seater 2.10x0.90x0.85, 28000; 2-seater 1.60x0.90x0.85, 19000 |
| 1 | tv_unit | standard 1.50x0.40x0.50, 9000; large 1.80x0.40x0.50, 12000 |
| 2 | coffee_table | standard 1.00x0.55x0.45, 6000 |
| 3 | bookshelf | standard 0.80x0.30x1.80, 7000 |
| 4 | armchair | standard 0.80x0.80x0.85, 8000 |
| 5 | side_table | standard 0.45x0.45x0.50, 2500 |

**Bedroom (K = 6 slots, P2):** 0 bed (queen 2.05x1.60 / single 2.00x1.00), 1 wardrobe (1.20x0.60), 2 nightstand_a (0.45x0.40), 3 nightstand_b (0.45x0.40), 4 desk (1.20x0.60), 5 desk_chair (0.50x0.50).

Sofa and tv_unit are mandatory for the living room; the rest are optional. The bed is mandatory for the bedroom.

Catalog flags: `coffee_table` has `rot_symmetry: 2`; `side_table` has `rot_symmetry: 4`; `nightstand_a` and `nightstand_b` form one interchangeable `group`; `tv_unit` has `needs_access: false` (a floor-standing unit placed against a wall that is viewed, not walked up to). An item with `rot_symmetry: 4` must be square in every variant, so reducing its rotation never changes its footprint; the loader checks this, along with slot order, positive sizes and unique names and ids.

Only the living room is in `configs/catalog.yaml` so far. The bedroom (P2) also needs heights, prices and the bed's facing, which the table above does not give.

### 2.2 Room and door sampling ranges (assumption)
| Room | Width `W` | Depth `D` |
|---|---|---|
| Living room | 3.5 to 7.0 m | 3.0 to 6.0 m |
| Bedroom | 3.0 to 5.0 m | 3.0 to 5.0 m |

Door wall uniform over 4 walls; `o` uniform in `[0, 1]`. The centre formula in Section 1 keeps the door centre at least 0.65 m, and the door edges at least 0.2 m, from any corner.

### 2.3 Layout vector

For slot `k = 0..K-1`:

| Symbol | Meaning | Type |
|---|---|---|
| `m_k` | Presence mask (given by user selection) | binary, input |
| `(w_k, d_k)` | Item size from catalog | float, input |
| `(u_k, v_k)` | Normalized centre | float in (0,1), target |
| `r_k` | Rotation class | categorical (4), target |

**Condition vector `c`** (dimension `7 + 3K = 25` for K = 6):

| Part | Dim | Encoding |
|---|---|---|
| Room size | 2 | `W / 8`, `D / 8` (fixed reference scale 8 m) |
| Door wall | 4 | one-hot in the order N, E, S, W (the order of the rotation classes) |
| Door offset | 1 | `o` |
| Presence masks | K | `m_k` |
| Item widths | K | `w_k / 3` (0 for absent) |
| Item depths | K | `d_k / 3` (0 for absent) |

**Target vector `x`** (dimension `6K = 36`): for each slot, `(u_k, v_k)` and one-hot `r_k` (4), slot after slot, so `x` reshapes to `(K, 6)`. Absent slots are zero-filled and masked out of the loss. Rotation targets are the canonical form defined in Section 1. Decoding takes the arg-max of the rotation logits and reduces it to canonical form; an encode-then-decode round trip returns the canonical layout (within 1e-5 m in float32).

### 2.4 Dataset generation

**Set A (good layouts, trains the CVAE):** 30,000 layouts per room type **(start)**. Settings in `configs/default.yaml` (`generator`); relation ranges from `configs/rules.yaml`.
1. Sample `W, D` and the door (Section 2.2), then the furniture: mandatory items always; each optional item with a probability that rises linearly with floor area, from 0.25 at 10.5 m² to 0.75 at 30 m² and above **(assumption)**. Small rooms average about one optional item and large rooms about three, and every combination still occurs at every size. Variants are equally likely.
2. Each placement attempt picks a **layout style** uniformly among those whose anchors fit the room, then the wall behind the sofa uniformly among the walls where that style fits:
   - (a) **wall sofa**: back against a wall, TV unit against the opposite wall. It fits only where the front-to-front distance lands in the sofa-TV range, an axis of 2.9 to 4.8 m for the catalog's depths, which rules it out in about a quarter of rooms;
   - (b) **floating sofa**: TV unit against a wall, the sofa facing it 1.5 to 3.5 m away with at least 0.75 m of walkway behind it (an axis of at least 3.6 m);
   - (c) **L-shape**: as (a), with the armchair's back against a side wall, facing the coffee table (needs both items).
3. Place the items with rule-driven anchors in the frame of the wall behind the sofa. Items against a wall stand 0 to 5 cm from it; every spacing (sofa to TV unit, coffee-table gap, side-table gap; armchair 0.4 to 0.9 m from the coffee table's end) is drawn uniformly inside its range; lateral jitter has sigma 0.05 m. Every item placed relative to the sofa limits where the sofa can stand along its wall without an item leaving the room or entering the door zone, and the sofa's position is drawn from what is left. The bookshelf goes last, on any free stretch of wall with 0.6 m of clear floor in front of it. Every kept layout therefore scores 1.0 on alignment and relations.
4. Run the hard checker and keep the first layout that passes. The room and furniture stay fixed for up to 20 attempts, so the rooms in the data follow the designed distribution; a room that fails every attempt is dropped (0.7% of rooms in v1). **Log the rejection rate by room-area bin and by item count** and plot it (`reports/figures/dataset_v1_rejection.png`: 79% of attempts rejected at 8 to 12 m², 12% at 28 to 32 m²). Rejection still makes the most crowded small rooms rarer than designed; state this in the report.

**Set B (labelled layouts, trains the evaluator):** 60,000 layouts **(start)**, aimed at about 50% valid. Settings in `configs/default.yaml` (`set_b`).
- Half are generator layouts as they come (all valid).
- Half are perturbed copies of fresh generator layouts, a tenth of Set B per type: (i) position jitter with sigma 0.1, 0.3 or 0.6 m (a third each), applied to each item with probability 0.5 and to at least one; (ii) one or two items turned to a facing that looks different, centres kept; (iii) every item placed uniformly inside the room, facing a random way; (iv) forced overlap: one item's centre moved to a random point inside another item; (v) near-miss, half of each kind: one item slid along x or y into a neighbour until they overlap by an area drawn log-uniformly from 0.002 to 0.0125 m², so half fall under the 0.005 m² tolerance; or one item slid to between 3 cm clear of the door clearance zone and 3 cm inside it. A near-miss move touches nothing else and keeps every item reachable, so the boundary is the only thing a checker has to judge. Each Set B sample stores its perturbation type so results can be reported per type.
- Labels are computed by the checker: `valid in {0,1}` and `quality in [0,1]`.
- Dataset v1 is 62% valid: clean 100%; jitter 47, 31 and 23%; rotation 29%; random 12%; forced overlap 0%; near-miss 49% (overlap) and 50% (door). It is above the 50% aim because some perturbations leave a layout valid; weight the classes in the loss if the imbalance matters.

**Outlier variant (for E2):** Set A plus 2 to 5% deliberately odd layouts (extreme positions), flagged in metadata.

**Splits:** 70 / 15 / 15 (train / validation / test) of Set A and of Set B, split by layout. Further test sets of **1,000 rooms each** (one generator layout per room as a reference) are **sampled separately** with the same generator, restricted to their region. Set A and Set B draw a room again whenever it falls in one of these regions, so training never sees them (asserted in a test):
- **Interpolation set:** room areas in [22, 26] m² **(start)**. About 21% of rooms would fall here under uniform `W` and `D`, so this leaves a visible gap in the training distribution.
- **Unseen-combination set:** room areas above 32 m² **(start)**, about 12% of rooms. Each individual `W` and `D` value still lies inside the training range, so this tests unseen *combinations*, not true extrapolation.
- **Out-of-range set:** `W` in (7.0, 8.0] m and `D` in (6.0, 7.0] m, outside every training range (the 8 m raster canvas and the `W/8` scaling still fit). The app shows an out-of-distribution warning for such inputs.
- **Reference set for diversity:** for 200 rooms of the Set A test split, 20 procedural layouts each (G0), used for the diversity ratio in E1 (3,992 layouts in v1: a few runs find no layout).

Above 32 m² almost every generated layout uses the floating sofa (99% of the unseen-combination set, all of the out-of-range set), because a wall sofa needs an axis of at most 4.8 m. E10 on those sets therefore mostly tests one style; say so when reporting it.

**Real-room set (P1):** 15 to 20 measured real rooms with furniture positions, never used for training.

**Storage:** `python run.py data` writes `data/v1/` in about 7 minutes: `.npz` arrays in meters (one row per layout, canonical form), CSV files with per-layout information (style, labels, quality terms) and the generator's attempt log, the split indices, the held-out and diversity sets, the `f_max` calibration rooms, and `metadata.json` (seed, the three config files, git commit, counts, label shares, `f_max`, a SHA-256 hash per file and one dataset hash). Each part draws from its own random stream spawned from the seed, and the hash covers the arrays inside the `.npz` files rather than the files (whose zip timestamps change), so a rebuild from the same commit reproduces it exactly. Rasters are produced on the fly.

---

## 3. Rule engine (`rules.py`)

### 3.1 Hard checks
| ID | Check | Definition | Tolerance |
|---|---|---|---|
| H1 | In-room | Each effective footprint lies inside `[0,W] x [0,D]` | 1e-3 m |
| H2 | No overlap | Pairwise intersection area of footprints | at most 0.005 m² **(start)** |
| H3 | Door clearance | Footprints do not intersect the door clearance zone | 0 area (1e-4 m²) |
| H4 | Reachability | Every item's access point reachable from the door (see 3.3) | n/a |
| H5 | Budget | Sum of selected catalog prices at most user budget | exact |

A layout is **valid** iff H1 to H4 hold (H5 is checked at selection time).

### 3.2 Overlap of two axis-aligned boxes
For boxes `i, j` with centres `(x, y)` and effective sizes `(w, d)`:

```
ox_ij = ReLU( min(x_i + w_i/2, x_j + w_j/2) - max(x_i - w_i/2, x_j - w_j/2) )
oy_ij = ReLU( min(y_i + d_i/2, y_j + d_j/2) - max(y_i - d_i/2, y_j - d_j/2) )
ov_ij = ox_ij * oy_ij            (area in m^2)
```

The exact area is what the **checker** uses. It is **not** suitable as a training or optimization loss: if one box's x-extent lies inside the other's (a side table beside a long sofa), `ox_ij` equals the smaller width and does not change when either box moves, so its gradient is 0 and a fully contained item gets no push at all.

For **latent optimization** use penetration depth instead, with clearance margin `mu = 0.05 m` **(start)**:

```
px_ij   = ReLU( (w_i + w_j)/2 + mu - |x_i - x_j| )
py_ij   = ReLU( (d_i + d_j)/2 + mu - |y_i - y_j| )
pen_ij  = min( px_ij, py_ij )            (0 if the boxes are separated on either axis)
L_ov    = sum_{i<j} pen_ij^2
```
`pen_ij` is the smallest shift that would separate the two boxes, so the gradient pushes the pair apart along the easier axis, even under full containment. Replace `|t|` by `sqrt(t^2 + 1e-6)` so the gradient is finite when two centres coincide. It is exactly 0 there by symmetry, but exact ties do not occur in practice; if one does, add a random offset of about 1e-3 m. The same formula is used between each item and the fixed door clearance rectangle.

### 3.3 Reachability (H4)
1. Grid of 0.10 m cells covering the room.
2. Mark cells covered by any furniture footprint as blocked.
3. Compute the Euclidean distance from each cell centre to the nearest obstacle or wall; the implementation measures it exactly (point to box and point to wall, with the door opening counted as open), which is a distance transform without its grid error. A cell is **passable** if that distance is at least `min_path_width / 2`, with `min_path_width = 0.6 m` **(assumption)**. Because cells are sampled at their centres, a corridor narrower than 0.6 m always blocks, one at least 0.7 m wide always passes, and one in between depends on how it lines up with the grid.
4. **Start cell** = the cell at the centre of the door clearance zone (0.45 m in front of the door centre); the door opening itself is treated as free space. Do not start from the cell just inside the door: it is closer than 0.3 m to the wall and would never be passable, so every layout would fail H4. Because H3 keeps furniture out of the zone, the start cell is passable in every layout that passes H3.
5. Label the connected components of the passable mask with `scipy.ndimage.label` (4-connectivity). The reachable region is the component that contains the start cell.
6. Each item has an **access line**: its whole front face moved 0.35 m out. The item is reachable if a cell of the reachable region lies within 0.3 m of its access line. (Up to v1.3 this was a single access point in front of the centre of the face. That fails every sofa with a coffee table in front of it: the nearest reachable cell is about 0.85 m from the point, although both ends of the sofa are open.) For items with `rot_symmetry > 1` there is one access line on **every equivalent face** (two for `rot_symmetry = 2`, four for `rot_symmetry = 4`), and the item counts as reachable if **any** of them is, because the canonical rotation is arbitrary among equivalent faces. Otherwise a side table stored at `r = 0` and pushed against the north wall would fail H4, and a coffee table with a sofa in the narrow gap on its stored front side would fail although its other side is open. Items with `needs_access: false` in the catalog (for example the `tv_unit`, a floor-standing unit viewed from a distance) are skipped.
7. The **reachability ratio** = reachable items / items that need access; H4 requires 1.0.

### 3.4 Quality score `S in [0,1]` (soft)
```
S = w_a*S_align + w_r*S_rel + w_c*S_circ + w_s*S_space
```
Default weights `(0.30, 0.30, 0.25, 0.15)` **(assumption, configurable)**.

| Term | Definition |
|---|---|
| `S_align` | Fraction of the items that should sit against a wall (living room: TV unit and bookshelf; the sofa is exempt because style (b) floats it) which are within 0.10 m of a wall with their back to it (orientation is checked only for items with `rot_symmetry = 1`; symmetric items are judged by position only) |
| `S_rel` | Mean of the relation scores that apply; a relation is skipped when one of its items is absent. Living room **(assumption)**: sofa and TV unit face each other with the TV unit within the sofa's width, front faces 1.5 to 3.5 m apart; the coffee table in front of the sofa, 0.35 to 0.50 m from its front face; the side table beside an end of the sofa, at most 0.15 m from it; the armchair at 90 degrees to the sofa, facing the coffee table, at most 1.2 m from it. A relation scores 1 inside its range, falls linearly to 0 at 0.5 m outside it, and is 0 when the arrangement itself is wrong (for example the TV unit facing away) |
| `S_circ` | Fraction of free floor cells that are passable (walkability). Known issue: cells within 0.3 m of a wall are never passable, so even an empty room scores only 0.68 (3.5 x 3.0 m) to 0.83 (7 x 6 m), which favours large rooms in comparisons across rooms (open decision T8, Section 11) |
| `S_space` | 1 while furniture area / room area lies inside 0.15 to 0.40, falling linearly to 0 at 0.15 outside the band (so 0 for an empty room and at 55% coverage) **(assumption)** |

All distance and angle ranges live in `configs/rules.yaml` and are documented as our assumptions.

### 3.5 Feasibility pre-check
Reject early if `sum(footprint areas) > f_max * (room area - door clearance area)` with **`f_max = 0.38`**, or if the cheapest selection exceeds the budget. Return a message with a suggestion (remove an optional item or choose a smaller variant).

`f_max` was calibrated on dataset v1 (T15). 3,903 rooms were spread evenly over the footprint ratio, with each optional item present with probability 0.5 whatever the room's size, because rooms drawn at random are almost never crowded. A logistic fit of "the generator furnished the room within 20 attempts" against the ratio crosses one half at 0.377 (`reports/figures/dataset_v1_f_max.png`). The starting value 0.45 was too generous: all six items (about 4.12 m²) would pass in a 10.5 m² room, where most such layouts fail the path rule. With 0.38 they are turned away below about 11.7 m² of floor. The calibration uses the generator's own styles, so a request it cannot furnish might still be possible with another arrangement; state this in the report.

---

## 4. Models

### 4.1 CVAE generator

**Purpose:** learn `p(x | c)`, the distribution of good layouts given the condition.

**Dimensions (K = 6):** `x` = 36, `c` = 25, latent `z` = 16 **(start)**.

**Encoder** `q(z | x, c)`
```
input  [x ; c]           (61)
Linear(61, 256) -> BatchNorm -> Act -> Dropout(0.1)
Linear(256, 256) -> BatchNorm -> Act -> Dropout(0.1)
Linear(256, 16) -> mu
Linear(256, 16) -> logvar
z = mu + exp(0.5*logvar) * eps,    eps ~ N(0, I)          (reparameterization)
```

**Decoder** `p(x | z, c)`
```
input  [z ; c]           (41)
Linear(41, 256) -> BatchNorm -> Act -> Dropout(0.1)
Linear(256, 256) -> BatchNorm -> Act -> Dropout(0.1)
head_pos : Linear(256, 2K) -> Sigmoid                       (u_k, v_k in (0,1))
head_rot : Linear(256, 4K) -> reshape (K,4)                 (logits; softmax only at inference)
```

`Act` defaults to ReLU; the final choice comes from experiment E3a (BatchNorm on, the deployed setting). The position head activation (Sigmoid vs Linear plus clamp) is tested in E7. *v1.5: Gate 2 keeps ReLU and the Sigmoid head, and freezes the MAE position loss (Section 11, `configs/frozen.yaml`).*

**Task to activation to loss mapping**

| Output | Task | Output activation | Loss |
|---|---|---|---|
| Position `(u, v)` | Bounded regression | Sigmoid (compare Linear + clamp) | MSE (default), MAE, Huber compared; **MAE frozen at Gate 2** |
| Rotation `r` | 4-class classification | Softmax (inside the loss; the network outputs logits) | Categorical cross-entropy via `F.cross_entropy` on the logits |
| Latent | Regularization | none | KL divergence to N(0, I) |

**Training loss (masked with a Hadamard product by `m_k`)**
```
L_recon = sum_k m_k * [ ell_pos( (u_k,v_k), (u_k*,v_k*) ) + lambda_rot * CCE(r_k) ]
KL      = -0.5 * sum_j ( 1 + logvar_j - mu_j^2 - exp(logvar_j) )
L       = mean_batch( L_recon ) + beta * mean_batch( KL )
```
- `ell_pos` is a sum over the two coordinates. `lambda_rot = 1` **(start)**.
- **KL annealing:** `beta` rises linearly from 0 to `beta_target` over the first 20 epochs **(start)**. The validation loss used for **early stopping and checkpoint selection is always computed with `beta = beta_target`**, and the patience counter starts only after annealing ends. Otherwise the rising `beta` makes the validation loss climb and stops training before annealing finishes.
- **`beta_target`:** sweep {0.01, 0.1, 0.5, 1.0} **(start)**. Because reconstruction errors on normalized coordinates are small, a large `beta` risks posterior collapse. Interpret the reconstruction term as a Gaussian log-likelihood with variance `sigma_x^2`, which effectively sets `beta`.
- **Optional physics-informed term (experiment):** add `lambda_ov * sum_{i<j} pen_ij^2` (penetration depth, Section 3.2) computed from decoded positions and ground-truth rotation. Toggle in E2 or E5.
- **Collapse diagnostic:** count *active latent units*, dimensions whose KL contribution exceeds 0.01 nats averaged over the validation set. If none are active, the decoder ignores `z` (collapse).

**Optimization defaults:** Adam, `lr = 1e-3`, batch 256, up to 200 epochs, early stopping on the `beta_target` validation loss with patience 15 (counted from epoch 21), weight decay 0 **(start)**. Keep the whole dataset on the GPU as tensors and index mini-batches directly (no `DataLoader`), so an epoch takes a fraction of a second.

**Parameter count:** roughly 0.2 M **(check in code)**.

### 4.2 CNN evaluator

**Input raster:** fixed physical canvas of 8 m x 8 m at 128 x 128 px (0.0625 m per pixel), so scale is preserved across rooms. The room sits at the canvas origin. Each pixel stores the **exact fractional area coverage** of each shape (anti-aliased), not a 0/1 value, so sub-pixel positions matter and small overlaps change the values. **4 channels:**

| Channel | Content |
|---|---|
| 0 | Room mask (coverage of the room) |
| 1 | Furniture coverage, **summed over items and not clipped**, so an overlap shows up as values above single-item coverage |
| 2 | Front-face strip (a band 0.10 m thick inside each item's front edge, about 1.6 pixels; encodes orientation). For items with `rot_symmetry > 1` the strip is drawn on **all equivalent faces**, so the raster carries no arbitrary orientation for them |
| 3 | Door and door-clearance zone |

**Resolution caveat:** the checker tolerates 0.005 m² of overlap. That is about 1.3 pixels of area, but along a 0.9 m edge it is a strip roughly 6 mm wide, a tenth of a pixel. Fractional coverage makes such a strip visible in principle (it changes the pixel values by about 0.1), but it sits at the limit of what the CNN can learn. The checker stays the authority, and F1 is reported separately for near-miss samples (see Metrics).

**Architecture**
```
Conv(4,16,3,pad1)  -> BN -> ReLU -> MaxPool2    (16 x 64 x 64)
Conv(16,32,3,pad1) -> BN -> ReLU -> MaxPool2    (32 x 32 x 32)
Conv(32,64,3,pad1) -> BN -> ReLU -> MaxPool2    (64 x 16 x 16)
Conv(64,64,3,pad1) -> BN -> ReLU -> MaxPool2    (64 x 8 x 8)
Flatten (4096) -> Dropout(0.3) -> Linear(4096,128) -> ReLU
head_valid : Linear(128,1)        -> Sigmoid  (in loss: BCEWithLogits)
head_score : Linear(128,1)        -> Linear   (regression, target in [0,1])
```
About 0.59 M parameters.

**Implementation.** Do not precompute rasters: Set B at 4 x 128 x 128 in float32 would need about 16 GB. Rasterize on the fly on the GPU. All footprints are axis-aligned boxes, so the exact coverage of a box in a pixel is the outer product of two 1-D overlaps (its x-interval with each column, its y-interval with each row); a batch of 64 layouts takes a few milliseconds. The same computation is piecewise linear in the box positions, which makes the differentiable version needed for M3 cheap.

**Training time and hardware.** A reviewer measured evaluator training at 128 x 128, batch 256, on 42k samples: about 2,600 samples/s (16 s per epoch) on the RTX 3050 and about 300 samples/s (140 s per epoch) on the CPU, before the extra cost of rasterizing on the CPU. The evaluator therefore needs the GPU; on CPU, use a reduced setting (64 x 64 raster, fewer samples, about 12 epochs fit in 30 minutes). The CVAE, latent optimization, rules and app are small enough for CPU. Re-measure on your own machines on Day 1, and schedule the single GPU explicitly (Development_Plan Section 2).

Our own measurement (T01, `python run.py check-env`, deterministic mode, same model and batch): about 1,960 samples/s on the RTX 3050 6 GB laptop GPU (21 s per 42k-sample epoch, 84 epochs in 30 minutes, 1.4 GiB peak memory) and 100 to 300 samples/s on the CPU (140 to 430 s per epoch, varying from run to run). Rasterizing on the GPU takes 3.6 ms for 64 layouts and 5.7 ms for 256.

**Loss:** `L_eval = BCE(valid) + lambda_s * Huber_or_MSE(score)`, `lambda_s = 1` **(start)**. The score loss is applied on all samples (score is defined for invalid layouts too).

**Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix on valid vs invalid; Spearman correlation of predicted score against rule score. **Report F1 separately for each Set B perturbation type** (jitter levels, rotation change, random placement, forced overlap, near-miss); an overall F1 alone hides the hard cases.

**Role of the evaluator (honest statement).** The checker's rules and the rule score are exact, so the evaluator is **not needed for correctness**, and it is **not a speed-up**: reachability for 64 candidates takes roughly 10 to 30 ms on a laptop CPU, about the same as rasterizing and scoring them with the CNN on the GPU, and both are tiny next to the 5 s latency budget. Its purpose is therefore: (1) a controlled study of a CNN on raw layout images (E9); (2) an experiment on whether a learned score can act as a differentiable surrogate for the non-differentiable rule terms (reachability, alignment) inside latent optimization (variant M3, experiment E8b, P1; it needs testing because the optimizer can exploit a surrogate); (3) a learned ranker for the top 3, reported next to the exact rule score and their correlation, with the rule score as the reference; (4) a starting point that could later be fine-tuned on human preference ratings, which the rule score cannot capture.

**Baseline:** MLP on hand-crafted features (total overlap, out-of-room area, door-clear flag, minimum pairwise distance, reachability ratio, wall-contact count, relation errors). *Honest caveat:* these features encode the very rules that create the labels, so the MLP is expected to do very well. The comparison shows what raw-pixel learning costs (the classic "manual feature engineering vs learned features" trade-off), not that the CNN wins.

### 4.3 Baselines

| ID | Method | Description |
|---|---|---|
| B1 | Uniform random | Each item's centre uniform over the positions where its footprint lies inside the room (so B1 does not fail on the walls alone), rotation uniform over 4 classes |
| B2 | Statistical sampling | From Set A train data, per slot the empirical joint `(u, v, r)` grouped by door wall and room-size bin (3 x 3 bins of equal shares of the training widths and depths; pooled over sizes, then over door walls, when a bin has fewer than 20 examples). A drawn `(u, v)` gets N(0, 0.02^2) noise. Items are placed one after another in slot order and drawn again, up to 10 times, while one overlaps an item already placed or sticks out of the room; no neural network. Without the out-of-room redraw B2 fails mostly on H1, because positions taken as fractions of a larger training room put wall items through a smaller room's wall (20% vs 58% raw valid on v1; the redraw was chosen) |
| M1 | CVAE only | Sample `z ~ N(0, I)`, decode |
| M2 | CVAE + latent optimization | M1 followed by the procedure in 5.2 |
| M3 (P1) | CVAE + latent optimization + CNN surrogate | M2 with the extra term `- lambda_c * q_hat` (needs the differentiable raster) |
| G0 | Procedural generator (reference) | The Set A generator with rejection sampling. Not a learned model and valid by construction, so it is a reference and is **not part of the validity comparison** (goal G2 is limited to B1 and B2). Its raw valid rate is defined as its **acceptance rate per attempt**: in evaluation it runs single attempts, so its raw samples are attempts. Report its **cost per valid layout** (attempts and seconds) and its diversity (the reference for the diversity ratio) |
| G0-pin (P1) | Procedural generator with a pin | Run the generator, move the pinned item to the requested position, then filter with the checker (repeat until valid or a cap). The strongest non-neural baseline for E12 |

All methods report raw validity (no filtering) and yield after filtering, with identical numbers of samples per input.

---

## 5. Inference pipeline

### 5.1 Steps
1. Validate input; run feasibility pre-check.
2. Select catalog variants if the user left them open (rule-based selection under budget and footprint limits; not learned). If several combinations fit, pick the one with the largest total footprint that still fits (comfort proxy) **(assumption)**.
3. Build the condition vector `c`.
4. Sample `N = 64` latent vectors; decode to positions and rotation probabilities.
5. Set rotation = argmax (fixed thereafter), except that a pinned item with a requested facing gets that rotation.
6. Latent optimization (5.2), optional. Afterwards **snap each pinned item exactly onto its requested position** before any check runs.
7. Run hard checks H1 to H4 and discard failures. The checker is always the final authority.
8. Score survivors with the CNN evaluator (and compute rule quality score for display).
9. Select the top 3 (5.3).
10. Render and report metrics.

### 5.2 Latent optimization
The decoder is frozen and put in `eval()` mode (BatchNorm uses running statistics, dropout off). Only `z` is optimized, as a batch of 64 independent problems.

```
minimize over z (each candidate starts at z0 ~ N(0, I)):
L_c(z) = lambda_ov  * sum_{i<j} pen_ij^2              (penetration depth, Section 3.2)
       + lambda_rm  * sum_i out_i
       + lambda_dr  * sum_i pen(i, door_zone)^2
       + lambda_pin * sum_{k in P} || p_k - p_k* ||^2
       + lambda_z   * 0.5 * || z - z0 ||^2
       - lambda_c   * q_hat(z)                        (M3 only: CNN score)
```
- `out_i = ReLU(w_i/2 - x_i) + ReLU(x_i + w_i/2 - W) + ReLU(d_i/2 - y_i) + ReLU(y_i + d_i/2 - D)` (protrusion outside the room; uses effective sizes). A fully inside item has zero loss and zero gradient here, which is correct.
- The `lambda_z` term is an **anchor to the candidate's own starting point `z0`**, not a pull towards 0. `z0` is a draw from the prior, so it starts in a high-density region, and the anchor keeps each candidate near it. A `||z||^2` term would drag every candidate to the same point `z = 0` and destroy diversity.
- `P` is the set of pinned items (P1). A pin gives a requested centre `p_k*` in meters and, optionally, a facing (N, E, S, W). If a facing is given, the pinned item's rotation is **overwritten** with it before optimization; otherwise the decoder's argmax is kept, which could leave a pinned sofa facing the wall. After optimization each pinned item is **snapped exactly** onto its requested position, so the pin is honoured by construction; the `lambda_pin` term only makes the other items adapt to it while `z` moves.
- Optimizer: Adam, `lr = 0.05`, `T = 150` steps **(start)**; stop a candidate early when `L_c` is below a small threshold. Adam does not guarantee a monotone decrease, so success is judged by the final loss, not by every step.
- Rotation is fixed from the decoder's argmax (non-differentiable choice); only positions move.
- Weights `(lambda_ov, lambda_rm, lambda_dr, lambda_pin, lambda_z)` start at `(10, 10, 10, 20, 0.05)` **(start)**; tune on the validation set.

### 5.3 Top-3 selection with diversity
1. Sort valid candidates by evaluator score (descending).
2. Greedily add a candidate if its mean per-slot displacement from every already-chosen layout is at least `tau_div = 0.3 m` **(start)** (rotation mismatch counts as a fixed penalty of 0.5 m for that slot).
3. If fewer than 3 pass, relax `tau_div` by 25% up to two times, then return what exists with a note.

---

## 6. Experiments

**Protocol.** Vary **one factor at a time** around the default configuration (no full grids). Screen every setting with **one seed**, then re-run the default and the best one or two settings with **three seeds** and report mean and standard deviation. No significance claims are made from three seeds. This is roughly 65 short CVAE trainings for E2 to E7 instead of about 210 with full grids.

**Order of work.** The ablations E2 to E7 finish first. The configuration is then **frozen** (Gate 2, saved as `configs/frozen.yaml`), and the headline experiments E1, E8 and E10 are run on the frozen configuration. Earlier runs of E1 and E10 on the default configuration are only for debugging the pipeline.

| ID | Question | Variables | Metrics | Plot or table |
|---|---|---|---|---|
| E1 | Does the neural pipeline beat the baselines, and at what cost? | B1, B2, G0 (reference), M1, M2 (M3 if built) | RVR, overlap, reachability, quality, diversity ratio to G0, **cost per valid layout** (attempts and seconds), latency | Bar chart and main results table |
| E2 | Which position loss is best, and how do losses handle outliers? | MSE, MAE, Huber (delta in {0.01, 0.05, 0.1}); clean vs outlier dataset; with/without overlap penalty | Position error (median and mean), RVR | Training curves; analytic plot of loss and gradient vs error |
| E3a | Which hidden activation should the deployed model use? | Sigmoid, Tanh, ReLU, LeakyReLU(0.01), ELU, GELU; depth 2; **BatchNorm on** (the deployed setting) | Validation loss, RVR after decoding | Table; decides T1 |
| E3b | How do activations behave without normalization (vanishing gradients, dying units)? | The same six activations; hidden depth in {2, 6}; **BatchNorm off**; plus BatchNorm on for Sigmoid and ReLU at depth 6 as a reference | Validation loss, per-layer gradient norm (epochs 1, 10, 50), dead-unit percentage | Gradient-norm plot; dead-unit bar chart; BN-on vs BN-off comparison |
| E4 | Which optimizer converges best? | SGD, SGD + momentum (0.9), RMSProp, Adam; lr grid | Epochs to reach target loss, final loss | Convergence curves |
| E5 | KL weight and latent size trade-off | beta in {0.01, 0.1, 0.5, 1.0}; z-dim in {4, 16, 32} | Reconstruction error, RVR, diversity, active units | Trade-off plot |
| E6 | Regularization effect | BatchNorm on or off; dropout in {0, 0.1, 0.3}; early stopping vs fixed epochs | Train-validation gap, validation loss | Curves and table |
| E7 | Does Sigmoid saturation hurt near walls? | Sigmoid vs Linear + clamp position head | Position error for items with `u` or `v` below 0.05 or above 0.95 vs the rest | Grouped bar chart |
| E8 | How much does latent optimization help? | T in {0, 25, 50, 100, 200}; lambda settings | RVR, diversity, runtime | RVR vs T curve |
| E8b (P1) | Does a CNN surrogate help latent optimization? | M2 vs M3 (adds `- lambda_c * q_hat`) | RVR, quality, reachability, and signs of exploiting the surrogate (CNN score rises, rule score does not) | Table |
| E9a | Evaluator quality | CNN; class balance; perturbation types | Precision, recall, F1 overall and **per perturbation type**, AUC, confusion matrix, Spearman with the rule score | Tables and matrices |
| E9b (P1) | CNN vs feature MLP | CNN vs MLP on hand-crafted features | Same metrics | Comparison table |
| E10 | Generalization beyond the training distribution | In-distribution test vs interpolation vs unseen-combination vs out-of-range sets | RVR and quality for M1 and M2 | Grouped bars |
| E11 (P1) | Behaviour on real rooms | 15 to 20 measured rooms | Qualitative plus RVR | Figure gallery |
| E12 (P1, top priority of the P1 items) | Pinned furniture: the strongest case for the learned model | 1 or 2 pinned items; M2 vs **G0-pin** (generator, move the pinned item to its spot, filter), and vs B1 and B2 with the pinned item fixed and the rest sampled then filtered. **Results are split by which item is pinned** (sofa, TV unit, coffee table, side table, ...) | Validity after snapping, cost per valid layout, quality, and the pre-snap displacement of the pinned item (how far optimization left it from the requested spot). The pin itself is exact by construction, so "pin satisfaction" is not the metric | Table per pinned item |

**Why E3 has two parts.** The deployed model uses BatchNorm, so the activation is chosen with BatchNorm on (E3a), where ELU, GELU and LeakyReLU are tested in the setting they will actually run in. The BatchNorm-off study (E3b) is a diagnostic: BatchNorm keeps pre-activations near zero mean and unit variance, which hides both vanishing gradients and dying ReLU. The BN-on references test that expectation directly; if BN does mask the effects, that is itself a finding to explain in the viva.

**Failure-case analysis:** collect at least 10 failed or poor outputs across methods, classify the cause (overlap, door blocked, unreachable, collapse), and discuss.

**As built (v1.5).**
- **The freeze is chosen on validation rooms, with M2.** The screening's M1 check sampled test rooms, so it only shortlists. Three seeds of each shortlisted setting are compared with M2, the deployed method, on 200 Set A validation rooms, under a rule fixed in advance (`python run.py gate2`, `reports/gate2.md`). M1 proved a poor guide: Tanh nearly doubles M1's raw valid rate but lowers M2's.
- **E1 also reports the quality of the top 3 the pipeline shows.** For each room, the valid layouts are ranked three ways (by the evaluator, by the exact rule score, and at random), and the diverse top 3 is scored under each. This measures what the evaluator adds as a ranker (Section 4.2, purpose 3).
- **E10 also runs G0 on every set.** It shows how hard each set's rooms are by themselves. The larger held-out rooms turned out easier, not harder.
- **Timings come from an interleaved pass (v1.6).** The laptop's speed changes with its power and thermal state, so methods timed one after another do not compare. For E1, E8 and E12 every method samples each room in turn, and E10 reports no timings.
- **E12 pins come from generator layouts (v1.6).** The pinned item stands where it stands in one generator reference layout of the same room, with that facing, so a valid completion is known to exist. Results are split by the pinned item, 60 requests each.

---

## 7. Metric definitions

```
RVR            = (# raw samples passing H1..H4) / (# raw samples)
Mean overlap   = mean over samples of sum_{i<j} ov_ij     (m^2)
Reachability   = mean over samples of reachable_items / items_that_need_access
Diversity      = mean over valid pairs (a,b) for one input of
                 (1/|K|) * sum_{k in K} ( ||p_k^a - p_k^b||_2 + 0.5*[r_k^a != r_k^b] )
Precision      = TP / (TP + FP)      Recall = TP / (TP + FN)
F1             = 2 * Precision * Recall / (Precision + Recall)
Spearman       = Pearson correlation of rank(score_pred) and rank(score_rule)
Active units   = # { j : mean_val KL_j > 0.01 }
```
`K` is the set of slots present in both layouts. Layouts are compared **after canonicalization**: for interchangeable groups take the minimum over permutations, and count a rotation mismatch only for slots with `rot_symmetry = 1` (compare modulo the symmetry otherwise), so meaningless flips do not inflate diversity. **Diversity ratio** = Diversity(method) / Diversity(G0 reference set), both over valid layouts for the same condition. **Cost per valid layout** = (time spent on sampling, optimization and checking) / (number of valid layouts produced). For G0 the raw valid rate is its acceptance rate per attempt, and G0 is left out of the validity comparison. **Expect G0 to be cheaper per valid layout than M2** for requests without a pin, because M2 makes the same checker calls plus 150 decoder passes. The learned approach has to earn its place through pinned completion (E12), amortized sampling and differentiable repair, and we report the numbers whichever way they fall.

**Evaluation protocol (T16, `spacegen/evaluate.py`).** Every method gets the same 500 rooms (the 200 diversity-reference rooms of the Set A test split, then 300 more test rooms at random) and 64 raw samples per room (`configs/default.yaml`, `evaluation`). The diversity ratio is taken over the rooms where both the method and the G0 reference have at least two valid layouts. G0's outputs are valid by construction, so its overlap and reachability are not reported. Baselines on v1 (`reports/tables/baselines.csv`):

| Method | RVR | Mean overlap | Reachability | Quality (valid) | Diversity ratio | Cost per valid layout |
|---|---|---|---|---|---|---|
| B1 | 10.1% | 0.21 m² | 0.55 | 0.29 | 0.98 | 9.9 attempts, 20 ms |
| B2 | 57.8% | 0.002 m² | 0.70 | 0.55 | 1.02 | 1.7 attempts, 4 ms |
| G0 | 74.7% per attempt | n/a | n/a | 0.88 | 1.00 | 1.3 attempts, 2 ms |

B2 avoids collisions but not bad arrangements (quality 0.55 against G0's 0.88), so E1 must compare quality as well as validity. The diversity ratio barely separates these methods.

---

## 8. Math appendix (viva preparation)

Both team members should be able to derive these on paper.

**A. Forward pass shapes.** For batch `B`, input `X in R^{B x d_in}`: `H = act(X W + b)` with `W in R^{d_in x d_out}`, `b in R^{d_out}` (broadcast). Matrix product `X W` is valid iff columns of `X` equal rows of `W`.

**B. Sigmoid and BCE.** `sigma(z) = 1/(1+e^{-z})`, `sigma'(z) = sigma(z)(1-sigma(z)) <= 0.25`. With `L = -[y log p + (1-y) log(1-p)]`, `p = sigma(z)`: `dL/dz = p - y`. Saturation: for large `|z|`, `sigma'` is near 0, so gradients vanish (relevant to positions near 0 or 1).

**C. Softmax and categorical cross-entropy.** `p_i = e^{z_i} / sum_j e^{z_j}`, `L = -sum_i y_i log p_i`: `dL/dz_i = p_i - y_i`.

**D. MSE, MAE, Huber.** MSE `L = (yhat - y)^2`, gradient `2 (yhat - y)` grows with error (outlier-sensitive). MAE gradient is `sign(yhat - y)` (constant, not differentiable at 0). Huber with threshold `delta`: `L = 0.5 e^2` if `|e| <= delta`, else `delta (|e| - 0.5 delta)`; `dL/de = e` or `delta * sign(e)`.

**E. ReLU family.** `ReLU'(z) = 1` for `z>0`, else 0 (choose 0 at `z=0`, a subgradient). Dying ReLU: a unit with `z<0` for all inputs gets zero gradient. Leaky ReLU keeps slope `alpha`. ELU: `alpha (e^z - 1)` for `z<0`, derivative `alpha e^z`.

**F. Backpropagation (one layer).** For `a = act(z)`, `z = W h + b`: `dL/dW = delta h^T`, `dL/db = delta`, `dL/dh = W^T delta`, where `delta = dL/da * act'(z)`.

**G. Adam.** `m_t = b1 m_{t-1} + (1-b1) g_t`, `v_t = b2 v_{t-1} + (1-b2) g_t^2`, bias correction `mhat = m_t/(1-b1^t)`, `vhat = v_t/(1-b2^t)`, update `theta <- theta - lr * mhat / (sqrt(vhat) + eps)`.

**H. ELBO for the CVAE.** `log p(x|c) >= E_{q(z|x,c)}[log p(x|z,c)] - KL( q(z|x,c) || p(z) )`. Maximizing the right side is minimizing `L_recon + KL`; `beta` re-weights KL.

**I. KL between a diagonal Gaussian and N(0, I).** `KL = 0.5 * sum_j ( mu_j^2 + sigma_j^2 - 1 - log sigma_j^2 )`, same as the code form with `logvar = log sigma^2`.

**J. Reparameterization.** `z = mu + sigma * eps`, `eps ~ N(0,I)`. Then `dz/dmu = 1` and `dz/dsigma = eps`, so gradients flow through the sampling step to the encoder.

**K. Overlap: why area fails as a loss and penetration depth works.** The exact intersection width is `ox = ReLU(min(x_i + w_i/2, x_j + w_j/2) - max(x_i - w_i/2, x_j - w_j/2))`. If box `j` lies inside box `i` along x, the `min` and `max` both pick box `j`'s edges, so `ox = w_j` regardless of position and `d ox / d x = 0`. A fully contained item therefore receives no gradient. Penetration depth `px = ReLU((w_i + w_j)/2 + mu - |x_i - x_j|)` has `d px / d x_i = -sign(x_i - x_j)` wherever `px > 0`, which pushes the centres apart even under full containment. With `pen = min(px, py)` the push acts along the axis that separates the boxes fastest (the minimum translation). Where `pen = 0` (separated) the gradient is 0: nothing to fix.

**L. Latent optimization with an anchor.** Minimizing `L_c(z)` with `0.5 lambda_z ||z - z0||^2` is a proximal (trust-region) step: the constraint terms act as a negative log-likelihood and the anchor as a Gaussian prior centred on the starting point `z0`, which was itself drawn from `N(0, I)`. It is not the global `N(0, I)` prior, which would pull every candidate to the same point `z = 0`.

**M. Data scaling.** Dividing coordinates by room size is scalar multiplication per axis (`u = x/W`). It makes the target independent of absolute room scale and keeps inputs in a range that suits Sigmoid and gradient-based training.

**N. Hadamard masking.** The loss term `m_k * (...)` is element-wise multiplication by the presence mask; absent items contribute zero loss and zero gradient.

**O. BatchNorm and vanishing gradients.** BatchNorm rescales pre-activations to roughly zero mean and unit variance before the activation. For Sigmoid this keeps inputs where `sigma'` is near its 0.25 maximum, and for ReLU it keeps a share of units active. With BN on, the vanishing-gradient and dying-ReLU effects that E3 tries to show are weakened, so E3 runs without BN and uses a small BN-on reference.

---

## 9. Implementation details

### 9.1 Repository structure
```
spacegen/
  configs/            default.yaml, catalog.yaml, rules.yaml
  spacegen/
    geometry.py       boxes, rotation, overlap (NumPy and PyTorch versions)
    catalog.py        load and query catalog
    layout.py         one layout as arrays, canonical form, layout JSON
    rules.py          door, hard checks, reachability, feasibility ratio
    quality.py        quality score S
    generator.py      procedural generator (Set A styles, G0)
    perturb.py        Set B perturbations and labels
    splits.py         splits, held-out regions and sets, diversity reference
    build_dataset.py  dataset build, f_max calibration, metadata and hash
    raster.py         layout to 4-channel raster
    dataset.py        layout vectors, storage, tensors for training
    baselines.py      B1, B2, G0 (G0-pin later)
    evaluate.py       evaluation harness for any sampler
    models/cvae.py
    models/evaluator.py
    models/mlp_baseline.py
    train_cvae.py
    train_evaluator.py
    latent_opt.py
    pipeline.py       end-to-end inference, top-3 selection
    metrics.py
    viz.py            2D plots (3D optional)
  experiments/        run_e1.py ... run_e12.py, make_figures.py
  app/streamlit_app.py
  notebooks/          01_manual_backprop.ipynb (toy example)
  tests/
  reports/            figures/, tables/
  data/               generated datasets (gitignored, regenerate by seed)
  README.md, requirements.txt, run.py, Makefile
```

`run.py` is the task runner (`python run.py <task>`: test, check-env, generator-report, set-b-report, data, baselines, with more added per task); Windows has no `make`, so the Makefile only forwards to it.

### 9.2 Configuration
YAML files with a single `seed` and per-experiment overrides. Every training run writes its config, seed, git commit and dataset hash next to the checkpoint.

**Seed helper (deterministic mode).** One helper sets the Python, NumPy and PyTorch seeds and enables `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark = False` and the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`. On the same machine and software versions, two runs with the same seed then give bit-identical weights (a reviewer confirmed this for a CVAE-like MLP and the 4-conv CNN with BatchNorm, cross-entropy, BCE and Adam). Final checks on the same machine are therefore **exact-equality** checks. Across machines, compare within a stated tolerance: RVR within 2 percentage points and F1 within 0.02 **(start)**. If an operation has no deterministic implementation PyTorch raises an error; handle that case explicitly and document it.

### 9.3 Testing plan
| Test | Purpose |
|---|---|
| Overlap: known boxes with analytic area; symmetry `ov_ij = ov_ji`; NumPy vs PyTorch agree | Correct geometry |
| Overlap loss gradient: finite-difference check vs autograd, **including a fully contained box (must have a non-zero gradient)**; for coincident centres assert only that the gradient is finite (it is exactly 0 there) | Correct differentiability |
| Rasterizer: the GPU outer-product rasterizer equals a brute-force supersampled reference; summed coverage matches the analytic area; symmetric items get strips on all equivalent faces | Correct CNN input |
| Generator: 100% of Set A "good" layouts pass the checker | Trustworthy data |
| KL: closed form equals `torch.distributions.kl_divergence` | Correct loss |
| Manual backprop: hand-derived gradients of a toy 2-item network equal autograd to 1e-6 | Viva evidence |
| Reachability: an empty room passes; a blocked door, a walled-off item and a door near a corner behave as expected; the start cell is passable | Correct H4 |
| Reachability of symmetric items: a side table against the north wall passes; a coffee table with a sofa in the gap on one face passes if the opposite face is open | Access checked on all equivalent faces |
| Pipeline smoke test: end-to-end run with tiny models returns 3 layouts | Integration |
| Determinism: two training runs with the same seed give identical weights on the same machine | Reproducibility claim is exact |
| Door placement: for all 4 walls and `o` in {0, 0.5, 1}, the door lies inside the wall, its centre is at least 0.65 m and both door edges at least 0.2 m from the corners | Unambiguous door definition |
| Canonicalization: every generated layout is in canonical form; swapping nightstands or rotating a coffee table by 180 degrees gives the same encoded target | No contradictory labels |
| Training loop: early stopping never triggers before annealing ends; selection loss uses `beta_target` | Correct KL annealing |
| Latent optimization: final constraint loss is lower than the initial loss for at least 90% of candidates | Sensible acceptance test (Adam is not monotone) |

### 9.4 Tooling
Python 3.11 or newer (tested on 3.12.5; the pinned NumPy 2.4 and SciPy 1.17 need 3.11); PyTorch (CUDA build if it works, otherwise CPU); NumPy; SciPy (connected components for reachability, the logistic fit for `f_max`); scikit-learn (metrics); pandas; Matplotlib; Streamlit; PyYAML; pytest; ipykernel, nbformat and nbclient for the notebooks. Plotly for the optional 3D view. Versions are pinned in `requirements.txt`. Random seeds set for Python, NumPy and PyTorch.

### 9.5 Streamlit app
Sidebar inputs: room type, `W`, `D`, door wall and offset, furniture selection and variants, budget, number of candidates, latent optimization on or off, optional pinned item position and facing. Main area: top-3 layouts with metrics; a "Compare methods" tab (B1, B2, G0, M1, M2 on the same input); a "Training and results" tab showing saved figures. Export buttons for JSON and PNG.

---

## 10. Limitations and ethics (to state in the report)

- **Synthetic data:** the model can only reproduce our rule-based notion of a good layout. It does not learn real designer taste. Design rules and score weights are assumptions.
- **Simplified geometry:** rectangular rooms, one door, four rotations, no windows or built-ins.
- **Illustrative catalog:** prices and sizes are placeholders, not retail data.
- **Small-scale evaluation:** three seeds and a small real-room set give descriptive, not statistical, evidence.
- **Learned vs rule-based:** the procedural generator already produces valid layouts, so the case for the CVAE is fast sampling, differentiable repair, and completion around pinned items, not validity by itself. The evaluator is not needed for correctness (the rule score is exact); its value is the CNN study and its possible use as a differentiable surrogate.
- **Rasterization resolution:** near-miss violations at the checker's tolerance are close to the raster's resolution limit.
- **Rejection-sampling bias:** crowded small rooms are under-represented in the data.
- **Ethics:** no personal data; do not present outputs as professional architectural or safety advice (for example, egress or accessibility regulations are not modelled).
- **Deployment:** local Streamlit for the demo. For real use, wrap inference in a REST API (for example FastAPI), version models and datasets, validate inputs, and monitor for out-of-distribution rooms (inputs outside the training ranges should trigger a warning).

---

## 11. Open technical decisions

| # | Decision | Default | How to decide | Outcome (v1.5, Gate 2) |
|---|---|---|---|---|
| T1 | Hidden activation | ReLU | E3a (BatchNorm on, depth 2) | ReLU. E3a ties ReLU and Tanh on validation loss; over three seeds Tanh lowers M2's raw valid rate (64% against 67%) and top-3 quality (0.72 against 0.79) |
| T2 | Position loss | MSE | E2 | **MAE**: M2 72% against 67% raw valid, top-3 quality 0.85 against 0.79, three seeds on validation rooms |
| T3 | Position head | Sigmoid | E7 | Sigmoid: no extra error near walls |
| T4 | `beta_target` | 0.1 | E5 | 0.1: `beta` 0.01 improves reconstruction but not M1's validity or quality |
| T5 | Latent size | 16 | E5 | 16: 4 and 32 change little |
| T6 | Optimization steps | 150 | E8 | 150: 63% at 100 steps, 64% at 200 on the default model; T33b reruns E8 on the frozen model |
| T7 | Bedroom in scope | No (P2); pinned furniture has priority | Development_Plan cut list | No; the team decides the extras at Gate 2 |
| T8 | Circulation term `S_circ` | Spec definition (passable over free cells; an empty room scores 0.68 to 0.83 depending on its size) | Before E10: divide by the same room's value when empty, so that every room can reach 1, if results are compared across room sizes | The spec's definition, kept. In E10, G0's quality moves only between 0.86 and 0.89 across the four test sets, and every set is compared with G0 |

---

## 12. Change log

**v1.1 (after technical review)**
- Overlap loss changed from intersection area to penetration depth (area has zero gradient under containment). Appendix K rewritten; test added.
- Reachability start moved to the centre of the door clearance zone; `scipy.ndimage.label` replaces the hand-written search.
- Early stopping and checkpoint selection now use `beta_target` and start after KL annealing.
- Raster changed to 128 x 128 with exact fractional coverage; F1 reported per perturbation type; near-miss perturbations added.
- Canonical labels for symmetric items and interchangeable slots; diversity computed after canonicalization.
- Door offset defined precisely (wall orientation, door centre, margin).
- E3 runs without BatchNorm (with a BN-on reference).
- Rotation head outputs logits (`F.cross_entropy`).
- Latent optimization anchor changed from `||z||^2` to `||z - z0||^2`; acceptance test changed from "monotone" to "final below initial".
- TV unit described as a floor-standing unit (`needs_access: false`).
- Added G0 (procedural generator) as a reference, cost per valid layout, and diversity ratio to G0 (replaces "not lower than B2").
- Held-out sets sampled separately and renamed; out-of-range set added.
- Experiments use one factor at a time with one-seed screening; headline runs happen after a configuration freeze.
- Rejection rate is logged by room area and item count; `f_max` is calibrated.
- Pinned furniture prioritised over the bedroom; bedroom moved to P2. CNN surrogate (M3, E8b) added as a P1 experiment.

**v1.2 (second technical review)**
- Symmetric items (`rot_symmetry > 1`): access checked on every equivalent face, front-face strip drawn on all equivalent faces, orientation excluded from `S_align`. Without this, valid layouts would fail H4 after canonicalization.
- Removed the claim that the CNN is a cheap pre-filter (measured: about break-even with the rule check). The evaluator is justified by the CNN study (E9a) and the surrogate experiment (E8b).
- G0 is excluded from the validity comparison (goal G2 covers B1 and B2 only), its raw valid rate is its acceptance rate per attempt, and the text now says G0 is likely cheaper per valid layout than M2. Added the G0-pin baseline for E12.
- E3 split into E3a (BatchNorm on, decides the deployed activation) and E3b (BatchNorm-off diagnostic). E9 split into E9a (P0) and E9b (P1).
- Corrected the pixel-size claim (0.005 m² is about 1.3 pixels of area, but a strip about a tenth of a pixel wide), the coincident-centre test (assert finite), the door distances (centre 0.65 m, edges 0.2 m), and the nightstand ordering (along the bed's lateral axis).
- Fixed leftovers: training overlap term uses penetration depth, reachability metric divides by items that need access, bedroom marked P2, G0 added to the repository structure and the compare tab.
- Rasters are built on the fly on the GPU (precomputed Set B would need about 16 GB).

**v1.3 (third technical review)**
- Pins: position and optional facing. The facing overwrites the pinned item's rotation before optimization, and the position is snapped exactly before the checks, so the pin is exact by construction. E12 metrics changed accordingly and results are split by which item is pinned.
- Hardware: the evaluator needs the GPU (measured 16 s vs 140 s per epoch); the CPU fallback holds for the CVAE only, with a reduced evaluator setting otherwise.
- Reproducibility: deterministic-mode seed helper (three settings) added; final checks are exact equality on the same machine, with a stated tolerance across machines.
- Leftover wording fixed (see PRD, Architecture and Development Plan change notes).

**v1.4 (implementation up to Gate 1: decisions taken while building T01 to T17)**
- Reachability (3.3): an item is reached through an **access line** along its whole front face, not a single point; the point failed every sofa with a coffee table in front. Distances are exact, and the corridor behaviour around 0.6 m is stated.
- Quality score (3.4): the four living-room relations and their ranges are defined, with a 0.5 m linear falloff and relations skipped when an item is absent; `S_space` falls to 0 at 0.15 outside its band; the sofa is exempt from `S_align`. The ceiling of `S_circ` (0.68 to 0.83 for an empty room) is recorded as open decision T8.
- Catalog (2.1): `rot_symmetry: 4` items must be square. Only the living room exists; the bedroom (P2) needs heights, prices and the bed's facing, and interchangeable groups are canonicalized only once it is built.
- Encoding (2.3): door one-hot order N, E, S, W; `x` slot after slot; decoding reduces the rotation arg-max to canonical form.
- Set A generator (2.4): the three styles are defined with the rooms they fit (a wall sofa needs an axis of 2.9 to 4.8 m); optional items appear with probability 0.25 to 0.75 by floor area; anchors in the frame of the wall behind the sofa, whose position is drawn from where every dependent item fits; the room stays fixed for up to 20 attempts. Every Set A layout scores 1.0 on alignment and relations.
- Set B (2.4): exact shares per type; the overlap near-miss range is 0.002 to 0.0125 m², log-uniform (was 0.002 to 0.05 m², which would make at least 70% of them invalid); door near-misses within 3 cm of the zone; near-miss moves touch nothing else and keep every item reachable. v1 is 62% valid.
- Splits and storage (2.4): held-out regions are excluded from Set A and Set B by drawing rooms again; held-out sets of 1,000 rooms; per-part random streams and a content hash that a rebuild reproduces exactly.
- Feasibility (3.5): `f_max = 0.38`, calibrated on rooms spread over the footprint ratio (logistic crossing at 0.377).
- Raster (4.2): front band 0.10 m; own hardware measurements added (21 s per epoch on the GPU, 140 to 430 s on the CPU).
- Baselines (4.3): B1 keeps items inside the room; B2 is specified, including redraws while an item overlaps another or sticks out of the room (20% vs 58% raw valid); G0 runs single attempts in evaluation.
- Metrics (7): evaluation protocol (500 rooms, 64 samples each) and the v1 baseline table.
- Tooling (9): Python 3.11 or newer; `run.py` task runner; notebook packages; repository structure updated.

**v1.5 (Week 2 built, Gate 2 review: decisions taken while building T19 to T30, and the frozen configuration)**
- CVAE (4.1): 177,732 parameters. The validation loss uses a fixed draw of `z`, so epochs compare. A linear position head is clamped only when generating. The optional overlap term uses margin 0, because training layouts may touch; E2 tries weight 1.0.
- Evaluator (4.2): 585,682 parameters, 19 s per epoch on the GPU. The trainer keeps the best validation epoch, because single epochs spike.
- Latent optimization (5.2):
  - Rotations stay those decoded at `z0`; a pin's facing overrides them.
  - A candidate stops once its constraint terms, without the anchor, fall below 1e-4.
  - It runs on the CPU, which is faster than the GPU for 64 candidates.
  - The 90% acceptance test counts the candidates that need repair. 11% already meet the constraints at `z0` and keep it; 99.6% of the rest end with a lower loss.
- Experiment details (6):
  - E2's outlier variant moves one item to a random spot in 4% of the training layouts.
  - E3b runs a fixed 60 epochs, so gradient norms exist at epochs 1, 10 and 50.
  - E4's target is a validation loss of 0.65.
  - E6's gap scores the training split in eval mode. Logged in train mode, dropout makes the training loss look worse than the validation loss.
  - E8 uses 100 test rooms.
- E1 and E10 additions (6): E1 adds the quality of each room's diverse top 3, under the evaluator's ranking, the rule score's and a random order. E10 adds G0 on every set as the reference.
- Configuration freeze (6, order of work):
  - The screening's M1 check sampled test rooms, so it only shortlists.
  - Three seeds of each shortlisted setting (Tanh, the overlap term, MAE, and the two best together) are compared with M2 on 200 Set A validation rooms, under a rule fixed in advance (`reports/gate2.md`).
  - MAE is frozen (`configs/frozen.yaml`). Tanh, which nearly doubles M1's validity, lowers M2's.
- Decisions T1 to T8 closed (11). T8 keeps the circulation term as specified.
- The PRD's raw-valid target ("the CVAE beats B1 and B2") is met by M2, not by M1: in the first pass of E1, M1 reaches 11% and M2 64%, against B2's 58%. Reported as measured.

**v1.6 (Week 3: final runs, pinned furniture, the app, regeneration)**
- Headline runs (6, 7): M1 and M2 are the mean and standard deviation over three seeds of the frozen configuration. Every seed samples the same rooms with the same draws, so the spread is the training's. B1, B2 and G0 do not depend on the CVAE and run once. E8's headline run uses the 200 diversity-reference rooms.
- Timing (7): cost per valid layout and seconds per room come from an interleaved timing pass on 100 rooms, in which every method, seed and step count samples each room in turn. Reason: the same M2 run was 1.6 times faster on one evening than on the afternoon before, and inside one E8 run the machine slowed down by half. Each command also times a fixed workload before and after. E10 has no timing columns.
- Pinned furniture (5.1, E12):
  - A pin is a centre in meters with an optional facing. It is refused when the item is not requested, or when on its spot it would stick out of the room or stand in the door's clearance zone.
  - M2 pulls the item to its pin while optimizing (`lambda_pin`) with the facing fixed, then snaps it. M1 snaps its sample. G0-pin moves the item of a generator layout onto its spot. B1 and B2 place the pinned item first, and B2 redraws the others against it.
  - Reported per pinned item: the raw valid rate after the snap, the share of requests with a valid layout, quality, the distance from the pin before the snap, and cost.
- Failure analysis: a valid layout scoring below 0.5 counts as poor; a room whose valid layouts are closer than 0.5 m on average counts as collapsed; a sample's cause is the first hard check it breaks, and the share breaking each check at all is reported too.
- Screening (6): the screening tables were rebuilt with the fields added after their runs (the diversity of M1's samples, the training loss in inference mode). A run's M1 check is repeated on the device it was trained on, because the CPU and the GPU draw different numbers from one seed.
- Reproducibility (9.2): `python run.py all` runs every step in order and `python run.py check-regeneration` compares tables (outside their timing columns), model hashes, the dataset hash and the frozen configuration with the saved ones. With `SPACEGEN_OUTPUT` set, a regeneration writes to another folder and starts from nothing. The final models are committed in `checkpoints/`, and the app falls back to them.
- App (9.5): the pin is placed by two sliders across and up the room; the app opens without trained models and names the command that creates each; `spacegen/app_logic.py` holds what it computes.
  - Under each layout the two large numbers are its rule quality and the evaluator's score, because those differ between the layouts of a request; the cost and the floor covered depend on the furniture alone and stand in one line below.
  - The app draws small plans whose fonts keep their size (`plot_layout` with a width), and it is light with the figures' blue as its accent (`.streamlit/config.toml`). Both came from looking at screenshots of the running app, which are kept in `reports/demo/` as the demo's backup.
- Documents: `python run.py pdf` prints the report, the math appendix, the failure cases, the viva notes and the demo script to A4 PDF files (`reports/pdf/`) through a headless Edge or Chrome; MathJax typesets the formulas.
- Not built: the feature-MLP comparison (E9b), the surrogate M3 (E8b), the real-room test (E11, which needs measured rooms), the bedroom and the 3D view.
