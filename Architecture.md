# SpaceGen AI: Architecture

| Field | Value |
|---|---|
| Version | 1.5 (the system as built; Sections 3, 4.1, 7 and 8 updated after Gate 2, see Section 13) |
| Companion docs | PRD.md, Tech_Spec.md, Development_Plan.md |
| Diagram format | Mermaid (renders in GitHub, VS Code preview, and most Markdown viewers). Plain-text fallbacks are included for the key diagrams. |

---

## 1. Architecture at a glance

SpaceGen AI is a **local, modular Python application** with three layers:

1. **Data layer:** catalog, rules, procedural generator, rasterizer, datasets.
2. **Model layer:** CVAE generator, CNN evaluator, baselines B1 and B2, and the procedural reference G0 (with a pinned variant). The feature-MLP baseline (P1) was not built.
3. **Inference and UI layer:** latent optimization, checker, ranking, Streamlit app.

The design principle is **learn to propose, then verify**. Neural networks propose and score layouts. A deterministic rule engine verifies them. Gradient-based latent optimization connects the two by using the differentiable part of the rules to steer the generator.

---

## 2. System context

```mermaid
flowchart LR
    U[User / Examiner] -->|room, door, furniture, budget, pins| APP[Streamlit App]
    APP --> PIPE[Inference Pipeline]
    PIPE --> CKPT[(Model checkpoints)]
    PIPE --> CAT[(Catalog and rules configs)]
    APP -->|top-3 layouts, metrics, plots| U
    DEV[Developers] -->|make data / train / experiments| TRAIN[Training and Experiment Scripts]
    TRAIN --> DATA[(Datasets)]
    TRAIN --> CKPT
    TRAIN --> RES[(Results: tables, figures)]
    APP --> RES
```

Plain-text version:
```
User -> Streamlit App -> Inference Pipeline -> {Checkpoints, Catalog/Rules configs}
Developers -> Training/Experiment scripts -> {Datasets, Checkpoints, Results}
Streamlit App -> shows Results (figures, tables)
```

---

## 3. Component view

```mermaid
flowchart TB
    subgraph DataLayer[Data layer]
        CATALOG[catalog.py]
        GEOM[geometry.py]
        RULES[rules.py, quality.py<br/>hard checks, quality score]
        GEN[generator.py, perturb.py<br/>Set A, Set B]
        SPLIT[splits.py, build_dataset.py]
        RAST[raster.py]
        DS[dataset.py, layout.py]
    end
    subgraph ModelLayer[Model layer]
        CVAE[models/cvae.py<br/>train_cvae.py]
        EVAL[models/evaluator.py<br/>train_evaluator.py]
        BASE[baselines.py<br/>B1, B2, G0]
    end
    subgraph InferenceLayer[Inference and UI layer]
        LOPT[latent_opt.py]
        PINS[pins.py]
        PIPE[pipeline.py]
        LOGIC[app_logic.py]
        APP[app/streamlit_app.py]
    end
    subgraph Experiments[Experiments]
        HARN[metrics.py, evaluate.py]
        EXP[experiments/<br/>screening, gate2, headline, pinned,<br/>failure_cases, figures, regenerate]
    end
    CATALOG --> GEN
    GEOM --> RULES
    GEOM --> RAST
    GEOM --> LOPT
    RULES --> GEN
    GEN --> SPLIT
    SPLIT --> DS
    DS --> CVAE
    RAST --> EVAL
    DS --> EVAL
    DS --> BASE
    GEN --> BASE
    CVAE --> LOPT
    LOPT --> PIPE
    PINS --> PIPE
    RULES --> PIPE
    EVAL --> PIPE
    PIPE --> LOGIC
    BASE --> LOGIC
    LOGIC --> APP
    PIPE --> HARN
    BASE --> HARN
    HARN --> EXP
```

### 3.1 Module responsibilities

All modules are in `spacegen/` unless a folder is given.

| Module | Responsibility | Depends on | Owner (suggested) |
|---|---|---|---|
| `geometry.py` | Rotation handling, effective footprints, exact overlap area for the checker, penetration depth for the losses, one implementation for NumPy and PyTorch | none | A |
| `catalog.py` | Items, variants, sizes, prices and flags from `configs/catalog.yaml`, with checks | config | B |
| `layout.py` | The `Layout` object, canonical form, the layout JSON | catalog | A |
| `rules.py` | Door geometry, hard checks H1 to H4, reachability on a grid, the footprint ratio | geometry, catalog | B |
| `quality.py` | Quality score `S` and its four terms | rules | B |
| `generator.py` | The procedural generator (three styles), Set A, the G0 sampler's core | rules, catalog | B |
| `perturb.py` | Set B: perturbations including near-misses, labelled by the checker | generator | B |
| `splits.py` | 70/15/15 splits, held-out regions and sets, the diversity reference | generator | A |
| `build_dataset.py` | Dataset v1 with metadata and a content hash, `f_max` calibration, dataset figures | generator, perturb, splits | B |
| `raster.py` | Layout to a 4-channel 128 x 128 fractional-coverage image on a fixed 8 m canvas, on the GPU | geometry | A |
| `dataset.py` | Vectors `x` and `c`, batches of layouts, tensors on the GPU | layout | A |
| `models/cvae.py` | Encoder, decoder, masked reconstruction loss, KL, active units | dataset | A |
| `train_cvae.py` | KL annealing, early stopping, per-epoch logs, config overrides | cvae | A |
| `models/evaluator.py`, `train_evaluator.py`, `evaluator_report.py` | The CNN with its validity and score outputs, its training, the E9a report | raster | B |
| `baselines.py` | B1 uniform random, B2 statistical sampler, G0 wrapper | dataset, generator | A |
| `latent_opt.py` | The constraint loss and Adam on `z`, pins, stopping | cvae, geometry | A |
| `pins.py` | Pinned furniture: the pin, the snap, G0-pin, why a pin is refused | layout, rules | A |
| `pipeline.py` | Request checks, variant choice, the M1 and M2 samplers, checks, ranking, the diverse top 3 | all above | shared |
| `metrics.py`, `evaluate.py` | Raw valid rate, overlap, reachability, quality, diversity; the same rooms and samples for every method | rules | shared |
| `viz.py` | Floor plans and the data figures | geometry | B |
| `app_logic.py`, `app/streamlit_app.py` | What the app computes, and its user interface | pipeline, baselines | B |
| `provenance.py`, `seed.py`, `paths.py`, `config.py`, `env_check.py` | Run records (seed, configs, git commit, dataset hash), deterministic seeding, output folders, the machine check | none | shared |
| `experiments/screening.py`, `gate2.py`, `headline.py`, `pinned.py`, `failure_cases.py` | E2 to E8; the freeze; E1, E8 and E10; E12; the failure analysis | all above | shared |
| `experiments/figures.py`, `make_figures.py`, `regenerate.py`, `demo_assets.py`, `report_pdf.py` | Every figure from saved tables; the full regeneration and its check; the demo's pictures; the report and the other documents as PDF files | all above | shared |

---

## 4. Data flow

### 4.1 Training-time flow

```mermaid
flowchart LR
    CFG[configs: catalog, rules, seed] --> GEN[Procedural generator]
    GEN --> CHK[Rule checker<br/>labels valid + quality]
    CHK --> SETA[(Set A: good layouts)]
    CHK --> SETB[(Set B: labelled layouts)]
    SETA --> ENC[Vector encoding x, c]
    ENC --> TRV[Train CVAE]
    SETB --> RAS[Rasterize on the fly]
    RAS --> TRE[Train CNN evaluator]
    TRV --> CK1[(runs/cvae/RUN/model.pt)]
    TRE --> CK2[(runs/evaluator/RUN/model.pt)]
    SETA --> STAT[Fit B2 on the training split<br/>each time it is used]
```

*As built:* every run folder also holds `run.json` (seed, configs, git commit, dataset hash), `log.csv` and `summary.json`. B2 is fitted in under a second and is not stored. The feature MLP was not built.

### 4.2 Inference-time flow

```mermaid
flowchart TD
    IN[User input] --> VAL[Validate + feasibility pre-check]
    VAL -->|infeasible| MSG[Message with suggestion]
    VAL -->|ok| SEL[Select catalog variants under budget]
    SEL --> COND[Build condition vector c]
    COND --> SAMP[Sample 64 latent vectors z]
    SAMP --> DEC[Decode positions + rotation probs]
    DEC --> ROT[Fix rotation = argmax]
    ROT --> LO{Latent optimization on?}
    LO -->|yes| OPT[Adam on z: penetration, out-of-room, door, pins, anchor to z0]
    LO -->|no| CHECK
    OPT --> CHECK[Hard checks H1..H4]
    CHECK -->|fail| DROP[Discard]
    CHECK -->|pass| SCORE[CNN evaluator score + rule score]
    SCORE --> TOP[Top-3 with diversity threshold]
    TOP --> OUT[2D render + metrics + export]
```

Plain-text version:
```
input -> validate/feasibility -> select variants -> build c -> sample z (64)
      -> decode -> fix rotation -> [latent optimization] -> hard checks
      -> CNN score -> top-3 with diversity -> render + metrics
```

---

## 5. Model architectures

### 5.1 CVAE

```mermaid
flowchart LR
    X[x layout 36] --> CONCAT1((concat))
    C1[c condition 25] --> CONCAT1
    CONCAT1 --> E1[Linear 61-256, BN, Act, Drop]
    E1 --> E2[Linear 256-256, BN, Act, Drop]
    E2 --> MU[mu 16]
    E2 --> LV[logvar 16]
    MU --> Z((z = mu + sigma * eps))
    LV --> Z
    Z --> CONCAT2((concat))
    C2[c condition 25] --> CONCAT2
    CONCAT2 --> D1[Linear 41-256, BN, Act, Drop]
    D1 --> D2[Linear 256-256, BN, Act, Drop]
    D2 --> HP[Position head: Linear 256-12, Sigmoid]
    D2 --> HR[Rotation head: Linear 256-24, logits per slot]
```

Notes:
- The condition `c` is concatenated at both encoder and decoder inputs.
- At generation time the encoder is unused: `z ~ N(0, I)` goes straight to the decoder.

### 5.2 CNN evaluator

```mermaid
flowchart LR
    R[Raster 4x128x128 fractional coverage] --> C1[Conv 16, BN, ReLU, Pool]
    C1 --> C2[Conv 32, BN, ReLU, Pool]
    C2 --> C3[Conv 64, BN, ReLU, Pool]
    C3 --> C4[Conv 64, BN, ReLU, Pool]
    C4 --> F[Flatten 4096, Dropout 0.3]
    F --> FC[Linear 128, ReLU]
    FC --> HV[Valid head: sigmoid, BCE]
    FC --> HS[Score head: linear, MSE or Huber]
```

### 5.3 Tensor shapes (batch B, K = 6)

| Tensor | Shape |
|---|---|
| Layout target `x` | (B, 36) |
| Condition `c` | (B, 25) |
| Latent `z`, `mu`, `logvar` | (B, 16) |
| Decoder positions | (B, 12) reshaped to (B, 6, 2) |
| Decoder rotation logits | (B, 24) reshaped to (B, 6, 4) |
| Presence mask `m` | (B, 6) |
| Raster | (B, 4, 128, 128) |
| Evaluator outputs | valid (B, 1), score (B, 1) |

---

## 6. Runtime sequence (app request)

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant P as Pipeline
    participant M as CVAE (frozen)
    participant O as Latent Optimizer
    participant R as Rule Checker
    participant E as CNN Evaluator
    User->>UI: room, door, furniture, budget, pins
    UI->>P: request
    P->>P: validate and feasibility check
    P->>M: sample 64 z, decode
    M-->>P: positions + rotation
    P->>O: refine z (150 Adam steps)
    O->>M: forward / backward through decoder
    O-->>P: refined layouts
    P->>R: hard checks H1..H4
    R-->>P: valid subset
    P->>E: score valid layouts
    E-->>P: scores
    P->>P: top-3 with diversity
    P-->>UI: layouts + metrics
    UI-->>User: 2D plots, scores, cost, export
```

---

## 7. Key data structures

### 7.1 Layout JSON (interchange format between pipeline, UI, exports and the 3D view)
```json
{
  "room": {"type": "living_room", "width": 5.2, "depth": 4.1},
  "door": {"wall": "S", "offset": 0.30, "width": 0.9},
  "items": [
    {"slot": 0, "id": "sofa_3seater", "w": 2.10, "d": 0.90, "h": 0.85,
     "x": 2.60, "y": 0.75, "rotation": 0, "price": 28000}
  ],
  "metrics": {"valid": true, "quality": 0.82, "cost": 43000, "floor_use": 0.24,
              "alignment": 1.0, "relations": 0.9, "circulation": 0.6, "space": 1.0},
  "meta": {"rank": 1, "method": "M2 (CVAE + latent optimization)", "seed": 0, "cvae": "runs/cvae/frozen/seed-0"}
}
```
(`rotation` uses the class index 0 to 3 defined in the Tech Spec. `room`, `door` and `items` are fixed (`spacegen/layout.py`); `metrics` and `meta` are free-form and shown here as the app writes them.)

### 7.2 Dataset file layout
```
data/v1/
  set_a.npz  set_a.csv  set_a_attempts.csv     # good layouts; per-layout style and quality; the generator's attempt log
  set_b.npz  set_b.csv                         # labelled layouts; perturbation type, valid, quality
  splits.npz                                   # row indices: set_a_train / validation / test, set_b_...
  held_out_interpolation.npz  held_out_unseen_combination.npz  held_out_out_of_range.npz   (+ .csv)
  diversity_reference.npz  diversity_reference.csv   # 20 generator layouts for each of 200 test rooms
  calibration.csv                              # the rooms of the f_max calibration
  metadata.json                                # seed, configs, git commit, counts, f_max, file hashes, dataset hash
```
Each `.npz` holds layouts in meters, one row per layout in canonical form: room, door, presence mask, variants, centres and rotations. The vectors `c` and `x` are computed from them when needed (`spacegen/dataset.py`).

---

## 8. Cross-cutting concerns

| Concern | Approach |
|---|---|
| **Configuration** | YAML files; one master seed; per-experiment overrides; config saved next to every checkpoint |
| **Reproducibility** | Seeds for Python, NumPy, PyTorch, with PyTorch in deterministic mode (bit-identical reruns on one machine); the dataset hash in every run record; `python run.py all` regenerates everything and `python run.py check-regeneration` compares it with the saved results |
| **Device handling** | `device = cuda if available else cpu`; batch sizes small so CPU works |
| **Logging** | Console plus CSV logs per run (loss curves, per-layer gradient norms, dead-unit percentage) |
| **Error handling** | Input validation with clear user messages; infeasible-request message; fallback when fewer than 3 valid layouts exist (return what exists, note it) |
| **Testing** | pytest suite (`python run.py test`), including a headless run of the app and of every experiment runner on a tiny dataset |
| **Timing** | The laptop's speed changes with its power and thermal state, so methods are timed in turn on the same rooms (`experiments/headline.py`) |
| **Out-of-distribution guard** | Warn if room size is outside the training ranges |
| **Security and privacy** | Local only; no personal data; inputs are numeric and validated |
| **Performance** | Batch all 64 candidates as tensors; vectorized pairwise overlap; reachability on small grids; training data kept on the GPU as tensors (no DataLoader); rasters built on the fly on the GPU as outer products of 1-D overlaps (precomputing Set B would need about 16 GB) |

---

## 9. Deployment view

**Demo (in scope):** everything runs locally.

```
Laptop (Windows/Linux) -> Python venv -> streamlit run app/streamlit_app.py
```

**Production sketch (for the ethics and deployment section of the report, not built):**
```
Client -> FastAPI (input validation, versioned models) -> Inference pipeline
       -> model registry (checkpoints + dataset hash) -> monitoring (OOD inputs, latency)
```

---

## 10. Architecture decision records (ADRs)

| ID | Decision | Alternatives considered | Rationale | Trade-off |
|---|---|---|---|---|
| ADR-1 | Synthetic, rule-generated dataset | Download real dataset (3D-FRONT, other) | We need our own labels (valid, quality) and inputs; real sets are scarce, large, or heavy to set up; synthetic gives control for experiments | Model learns our rules; mitigated with several styles, jitter and a real-room test set |
| ADR-2 | Furniture sizes are inputs, model predicts centre and rotation only | Predict sizes too | Sizes are known from the catalog; smaller output; exact overlap math | Cannot suggest new sizes |
| ADR-3 | CVAE generator | GAN, diffusion, transformer | Small data vectors, stable training, fits laptop, clear math (ELBO, KL, reparameterization), diversity via sampling | Possible blurry or averaged outputs; posterior collapse risk |
| ADR-4 | One model per room type (config driven) | One model with union of slots | Simpler slot semantics, less confusion | Two trainings if bedroom is included |
| ADR-5 | Latent optimization after generation | Filtering only; penalty only at training | Uses backprop at inference; directly reduces violations; supports pinned items | Extra hyperparameters; can reduce diversity |
| ADR-6 | Rotation is fixed (argmax) during latent optimization | Relax rotation with soft probabilities | Rotation is discrete; keeping it fixed avoids non-differentiable steps | Cannot fix a wrongly rotated item by optimization; a pinned item is the exception, because the user's requested facing overwrites its rotation before optimization |
| ADR-7 | Fixed 8 m physical canvas at 0.0625 m per pixel (128x128) with exact fractional coverage | Stretch each room to a fixed size; 0/1 pixels at 0.125 m | Preserves real scale; sub-pixel overlaps change the pixel values, so near-misses near the checker tolerance are visible | Bigger input and slower CNN; near-misses are still at the resolution limit |
| ADR-8 | Rule engine is deterministic and separate from learning | Learn constraints only | Guarantees validity; gives labels; makes results verifiable | Rules are our assumptions |
| ADR-9 | Streamlit UI | Web stack (React plus API) | Fast to build in Python; enough for the demo | Not production-grade |
| ADR-10 | Keep the CNN evaluator as a study, a possible surrogate, and a learned ranker reported next to the exact rule score; do not present it as a speed-up | Rank by rule score only; drop the CNN | The CNN is not needed for correctness and is not faster than the rule check (both tiny next to the latency budget). It supplies the CNN study (E9a) and the surrogate experiment (M3, E8b); the rule score stays the reference | Extra components whose value is experimental and educational; stated plainly |
| ADR-11 | Penetration depth (min over axes) as the overlap loss; exact area only in the checker | Intersection area as the loss | Area has zero gradient when one box is contained in the other's extent; penetration depth always pushes the pair apart | Slightly different from the checker's measure, so validity is always judged by the checker |
| ADR-12 | Anchor term `norm(z - z0)^2` in latent optimization | Prior term `norm(z)^2` | The prior pulls every candidate to `z = 0` and destroys diversity; the anchor keeps candidates near their own start | Loses the global-prior interpretation; it acts as a trust region |
| ADR-13 | Canonical labels for symmetric items, with access and orientation handled over all equivalent faces | Arbitrary rotations; canonical labels with single-face checks | Arbitrary rotations give contradictory targets; single-face access would reject valid layouts (a side table stored at r=0 against the north wall) | Slightly more complex checker and raster |

---

## 11. Extension points

| Extension | Where it plugs in | Cost |
|---|---|---|
| Bedroom domain (P2) | New catalog and rules config; retrain CVAE with the same code | Low to medium |
| Windows | Add channel to raster, extra clearance rule, extra condition entries | Medium |
| GRU autoregressive decoder | Alternative `models/` module with the same interface as the CVAE | Medium |
| 3D view | Consumes layout JSON only; no model changes | Low |
| CNN surrogate in latent optimization (M3) | Differentiable raster in `raster.py`; extra term in `latent_opt.py` | Medium; risk that the optimizer exploits the surrogate |
| Natural-language input | New front-end module producing the same request object | Medium |
| Multi-room layouts | Would need new representation and models | High (out of scope) |

---

## 12. Design invariants (rules the code must respect)

1. All geometry checks run in **meters**; normalized coordinates exist only at the model boundary.
2. The **checker is the single source of truth** for validity; no model output is shown as valid without passing it.
3. Every "good" training layout is **re-verified** by the checker before it enters Set A.
4. Models never see held-out room sizes during training or model selection.
5. Every reported number can be regenerated from a **seed, config and dataset hash**.
6. During latent optimization, the decoder is frozen and in `eval()` mode.
7. Training targets are in **canonical form** (symmetric items and interchangeable slots), and metrics compare layouts after canonicalization. The checker and raster treat all equivalent faces of symmetric items alike.
8. Early stopping and checkpoint selection use the validation loss at `beta_target` and start only after KL annealing ends.

---

## 13. Change notes

**v1.5 (the system as built, after Gate 2)**
- Sections 3 and 3.1 list the modules that exist. The data layer gained `layout.py`, `quality.py`, `perturb.py`, `splits.py` and `build_dataset.py`; training, the evaluator's report, pins, the app's logic, run records and the experiment runners have their own modules.
- Not built: the feature-MLP baseline (P1), the differentiable raster and the surrogate M3 (P1), the 3D view (P2) and the bedroom (P2). Section 11 still describes where they would plug in.
- Section 4.1: B2 is fitted when used, not stored. Section 7 shows the dataset files and the layout JSON as written.
- Section 8: regeneration and its check, and why timings are measured with the methods taking turns.
- Invariant 4 holds as stated: the frozen configuration was chosen on validation rooms (`reports/gate2.md`).
