# SpaceGen AI — Deep Learning Mini Project Presentation & Viva Guide

> **Your Presentation Date:** Tomorrow  
> **Subject:** Deep Learning  
> **Project Name:** SpaceGen AI — Constraint-Aware Layout Generation using Conditional VAE and Latent Optimization

---

## 1. The 30-Second Elevator Pitch
*(Memorize this word-for-word for your opening introduction)*

> "Good morning, professors. Our Deep Learning mini project is **SpaceGen AI**.  
> The goal of our project is to solve **automated interior layout generation** using Deep Learning.  
> Given room dimensions and door locations, our system generates functional, non-overlapping furniture arrangements.  
> We use a **Conditional Variational Autoencoder (CVAE)** to generate diverse layouts, a novel **Latent Space Optimization** technique via gradient descent to repair overlapping furniture, and a **Convolutional Neural Network (CNN) Evaluator** to score and rank the best floorplans.  
> Everything is packaged into a real-time interactive demo."

---

## 2. The 3-Minute Slide-by-Slide Presentation Structure

If you have slides or are presenting orally, follow this exact 6-step flow:

```
[1. Problem & Challenge] 
       ↓
[2. Why Deep Learning? (Multimodality)]
       ↓
[3. Core DL Model: Conditional VAE (CVAE)]
       ↓
[4. Novel DL Innovation: Latent Space Optimization]
       ↓
[5. Ranking Model: CNN Evaluator]
       ↓
[6. Live Demo & Results]
```

### Slide 1: Problem Statement
- **Problem:** Arranging furniture in a room requires satisfying hard physical constraints (no overlaps, clear pathways, wall alignments).
- **Goal:** Build an end-to-end Deep Learning pipeline that takes room specifications and outputs valid, aesthetically pleasing furniture layouts.

### Slide 2: Why Deep Learning? (The Multimodality Problem)
- A room does not have a single "correct" answer; it has **many good layouts** (sofa on North wall OR sofa on South wall).
- A traditional regression network (plain MLP) would output the *average* of all layouts—placing a sofa nonsensically in the middle of the room!
- **Deep Learning Solution:** Generative modeling using a **Conditional VAE**, where a continuous latent vector $z \sim \mathcal{N}(0, I)$ samples different layout modes.

### Slide 3: Model 1 — Conditional VAE (CVAE) Architecture
- **Input:**
  - Condition $c \in \mathbb{R}^d$: Room width, depth, door location, and furniture item list.
  - Layout $x$: Normalized coordinates $(x, y) \in [0, 1]$ and 4-way rotation classes (North, East, South, West).
- **Encoder:** Multi-Layer Perceptron (MLP) mapping $[x, c] \to \mu, \log \sigma^2$.
- **Reparameterization Trick:** $z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ (enables backpropagation through stochastic sampling).
- **Decoder:** MLP mapping $[z, c] \to \hat{x}$ (predicts continuous $(x,y)$ via Sigmoid and 4-class rotation logits via Softmax).
- **Multi-Task Loss:**
  $$\mathcal{L} = \mathcal{L}_{\text{pos}} (\text{MAE}) + \mathcal{L}_{\text{rot}} (\text{Cross-Entropy}) + \beta \cdot D_{\text{KL}}(q(z|x,c) \,\|\, \mathcal{N}(0, I))$$

### Slide 4: Model 2 — Latent Space Optimization (The "Magic" Step)
- Raw VAE samples sometimes have small overlaps (17% valid initially).
- **How we fix it using Deep Learning:**
  - We **freeze** the trained decoder weights $\theta$.
  - We define a differentiable collision penalty loss $\mathcal{L}_{\text{collision}}$ between furniture bounding boxes.
  - We backpropagate the error **all the way back to the latent vector $z$**:
    $$z \leftarrow z - \alpha \nabla_z \mathcal{L}_{\text{collision}}$$
  - We take 20 gradient steps in latent space before final decoding.
- **Result:** Validity jumps from **17% to 70%** without any post-processing heuristics!

### Slide 5: Model 3 — CNN Layout Evaluator
- To rank layouts, we convert the generated coordinates into a **64x64 multi-channel 2D floorplan image**.
- A 4-layer **Convolutional Neural Network (CNN)** with BatchNorm and LeakyReLU processes spatial features (clearances, viewing angles to TV, circulation pathways).
- Outputs a continuous layout quality score $[0, 1]$ to pick the **Top 3** candidate designs for the user.

### Slide 6: Results & Metrics
- Evaluated on 500 test rooms across multiple seeds:
  - Raw CVAE Validity: **17.3%**
  - CVAE + Latent Optimization Validity: **69.8%** (4x improvement!)
  - CNN Ranking lifts top layout quality from **0.67 to 0.85**.
  - Generation latency: **< 3 seconds** on CPU.

---

## 3. Deep Learning Architecture Cheat Sheet (Formulas & Layers)

### 1. Reparameterization Trick
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
- *Why:* Direct sampling has no derivative $\frac{\partial z}{\partial \mu}$. Reparameterization separates the stochastic node $\epsilon$ from parameters $\mu, \sigma$, allowing end-to-end backprop.

### 2. Closed-Form KL Divergence for Gaussian Prior
$$D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)) = \frac{1}{2} \sum_{i=1}^d \left( \mu_i^2 + \sigma_i^2 - 1 - \ln(\sigma_i^2) \right)$$

### 3. Loss Scaling & Annealing
- $\beta$-annealing schedule: $\beta$ increases linearly from $0.0 \to 0.1$ over the first 20 epochs.
- *Why:* Prevents **posterior collapse** (where the encoder ignores $z$ and just outputs the prior early in training).

---

## 4. Top 8 Viva Questions & Exact Answers

### Q1: Why did you use a Conditional VAE (CVAE) and not a plain Neural Network?
**Answer:**  
> "A plain neural network is deterministic and predicts only one output for a given room. But interior layout is a **multimodal problem**—a living room can have multiple equally valid arrangements. A standard network trained on MSE would predict the average of all valid layouts, which causes furniture to float in the center of the room. A CVAE learns a continuous latent distribution $\mathcal{N}(0, I)$ where each latent sample $z$ decodes into a distinct, valid design."

### Q2: Why CVAE instead of a GAN (Generative Adversarial Network)?
**Answer:**  
> "VAEs provide a smooth, continuous latent space with explicit density estimation and do not suffer from **mode collapse** or training instability like GANs. Furthermore, because the VAE latent space is continuous and well-behaved, we can perform **Latent Space Optimization** via gradient descent on $z$, which is difficult in GANs."

### Q3: What is Latent Space Optimization? Are you retraining the network?
**Answer:**  
> "No, the network weights are frozen. Latent Space Optimization treats the input latent code $z$ as a learnable parameter. We pass $z$ through the frozen decoder to predict furniture coordinates, compute a differentiable collision loss, and backpropagate the gradient $\nabla_z \mathcal{L}$ back to $z$. Taking 20 gradient steps shifts $z$ along the manifold to a region where furniture does not overlap. It increases layout validity from 17% to 70%."

### Q4: What is the reparameterization trick and why is it needed?
**Answer:**  
> "Sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is a non-differentiable operation, which blocks gradients from flowing from the decoder back to the encoder. The reparameterization trick expresses $z$ as $z = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$ is an external random noise tensor. Now, $\mu$ and $\sigma$ are deterministic linear operations, so gradients can flow freely via standard backpropagation."

### Q5: What is Posterior Collapse and how did you prevent it?
**Answer:**  
> "Posterior collapse happens when the decoder learns to ignore the latent code $z$ and relies solely on the room condition $c$. The KL divergence drops to zero, and all latent dimensions become inactive. We prevented this using **$\beta$-annealing**: we set $\beta = 0$ for initial epochs so the network first learns good reconstructions, and then gradually warm up $\beta$ to 0.1. We tracked active latent units and confirmed 10 of 16 dimensions remain fully active."

### Q6: Why did you use MAE loss for positions instead of MSE?
**Answer:**  
> "In our ablation studies, MAE (Mean Absolute Error / L1 loss) outperformed MSE (L2 loss) by $3\times$ (0.19m coordinate error vs 0.58m). MSE gradients scale with error magnitude ($2e$), which produces very small gradients when items are close to their target positions. MAE provides a constant gradient magnitude $(\pm 1)$, keeping gradients balanced against the KL divergence term."

### Q7: Why do you need a CNN Evaluator if you already have the CVAE?
**Answer:**  
> "The CVAE decoder is an MLP operating on 1D vectors; it generates candidates quickly, but doesn't have 2D spatial awareness. The CNN Evaluator receives a **rasterized 2D multi-channel floorplan** image. Convolutional filters naturally capture spatial features like clearance corridors, visual alignment between sofa and TV, and door clearance. It scores each candidate to surface the top 3 best layouts."

### Q8: What are the exact inputs and outputs of your CVAE?
**Answer:**  
> "- **Condition $c$:** Room width, depth, door location wall, door offset, and furniture type presence vector.  
> - **Input Layout $x$ (for Encoder):** Normalized coordinates $(u, v) \in [0, 1]$ and 4-class one-hot rotation for each piece of furniture.  
> - **Latent Space $z$:** 16-dimensional continuous Gaussian vector.  
> - **Output $\hat{x}$ (from Decoder):** Sigmoid-activated coordinates (guaranteed to be inside room bounds) and Softmax logits for the 4 cardinal facing directions."

---

## 5. How to Run the Demo Tomorrow

We have created two simple 1-click launchers in the project root:

### Option 1: Full Interactive Web Application (Recommended for Demo)
1. Double click **`run_app.bat`** (or run `streamlit run app/streamlit_app.py` in `.venv`).
2. The Streamlit UI opens in your browser (`http://localhost:8501`).
3. **What to show the examiner:**
   - Adjust the **Width** (e.g., 5.0m) and **Depth** (e.g., 4.0m) sliders.
   - Choose furniture items: **Sofa**, **TV Unit**, **Coffee Table**, **Armchair**.
   - Show the generated **Top 3 Layouts** with clearance corridors and furniture orientations.
   - Show the **Latent Optimization** toggle: Turn it off to show how pure CVAE produces slight overlaps, and turn it on to show how gradient descent repairs them into clean layouts!

### Option 2: Quick Python CLI Demo (If you want to show pure code execution)
1. Double click **`run_demo.bat`** (or run `python quick_demo.py` in `.venv`).
2. It prints model loading, latent sampling, optimization, and saves `quick_top1.png` in 3 seconds!

---

## 6. Summary of Key Files (Scaled-Down View)

| File | Purpose |
|---|---|
| **`spacegen/models/cvae.py`** | The PyTorch CVAE model (Encoder, Reparameterization, Decoder). |
| **`spacegen/latent_opt.py`** | The PyTorch gradient descent optimization on latent vector $z$. |
| **`spacegen/models/evaluator.py`** | The 4-layer PyTorch CNN layout quality evaluator. |
| **`spacegen/pipeline.py`** | The end-to-end inference pipeline combining CVAE + Opt + CNN. |
| **`quick_demo.py`** | 1-file standalone test script for fast presentation. |
| **`app/streamlit_app.py`** | The interactive browser demo. |
| **`run_app.bat` / `run_demo.bat`** | 1-click Windows shortcuts. |
