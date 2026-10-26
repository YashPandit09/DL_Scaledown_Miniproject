# SpaceGen AI — Mini Project (Scaled-Down)

This folder contains a **simplified, self-contained version** of the project
that is easy to understand and explain to your professor.

---

## Project Structure

```
mini_project/
├── model.py      ← All Deep Learning models (CVAE + CNN Evaluator)
├── train.py      ← Training loop (with synthetic data, no dataset needed)
├── demo.py       ← Real inference using pre-trained checkpoints
└── output/       ← Generated floorplan PNGs saved here
```

---

## What Each File Teaches

### `model.py` — The Deep Learning Architecture

**1. CVAE (Conditional Variational Autoencoder)**

```
Encoder:    [layout x + condition c] → mu, logvar
                                               ↓
Reparam:    z = mu + sigma * epsilon    (epsilon ~ N(0,I))
                                               ↓
Decoder:    [z + condition c] → positions (Sigmoid) + rotations (Softmax)
```

Key concepts:
- **Reparameterization Trick** — makes z differentiable so gradients can flow to encoder
- **Sigmoid on positions** — guarantees furniture stays inside the room
- **Softmax on rotations** — 4-class classification: North / East / South / West

**2. CNN Evaluator**

```
4-channel image (64×64) → 4× [Conv → BatchNorm → ReLU → MaxPool]
    → Flatten → Dropout → Linear → ReLU
    → head_valid (BCE loss)   → P(layout is valid)
    → head_score (MSE loss)   → quality score [0, 1]
```

---

### `train.py` — The Training Loop

**CVAE Loss:**
```
Total Loss = MAE(positions) + CrossEntropy(rotations) + β × KL_Divergence
```

- **MAE** (not MSE): gives larger, more balanced gradients for position regression
- **Cross-Entropy**: for 4-class rotation prediction
- **KL Divergence** (closed form):  `−0.5 × Σ(1 + logvar − μ² − exp(logvar))`
- **β-Annealing**: β starts at 0 and ramps to 0.1 over 5 epochs to prevent posterior collapse

**CNN Evaluator Loss:**
```
Total Loss = BCE(valid/invalid) + MSE(quality score)
```

---

### `demo.py` — The Real Inference Pipeline

Uses the **actual pre-trained checkpoints** to generate layouts:

```
Step 1: Sample z ~ N(0, I) → CVAE Decoder → 64 candidate layouts
Step 2: Latent Optimization → gradient descent on z to fix overlaps  
Step 3: CNN Evaluator scores all valid layouts → Top 3 picked
```

---

## How to Run

```bash
# From the project root directory:

# Test model shapes (no data needed):
python mini_project/model.py

# Train on synthetic data (no dataset needed, ~30 seconds):
python mini_project/train.py

# Run real inference with pre-trained models:
python mini_project/demo.py
```

---

## Key Deep Learning Concepts to Explain

| Concept | Where | One-Line Explanation |
|---|---|---|
| Reparameterization Trick | `model.py` line ~60 | Moves randomness into ε so gradients can flow |
| KL Divergence | `model.py` line ~95 | Regularizes z toward N(0,I) to prevent mode collapse |
| β-Annealing | `train.py` line ~80 | Slowly increases KL weight to avoid posterior collapse |
| Sigmoid on positions | `model.py` line ~70 | Forces (x,y) ∈ [0,1], keeping furniture inside the room |
| Softmax for rotation | `model.py` line ~72 | 4-class classification: N/E/S/W is not a number |
| MAE vs MSE | `model.py` line ~87 | MAE has constant gradient ±1; better balance with KL term |
| Latent Optimization | `demo.py` | Gradient descent on z (frozen decoder) fixes overlaps |
| CNN for ranking | `model.py` line ~110 | Conv layers capture spatial features (clearance, alignment) |
