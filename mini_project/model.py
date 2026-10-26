"""
SpaceGen AI - Deep Learning Models (Simplified for Mini Project)

Two models used in this project:

1. CVAE (Conditional Variational Autoencoder)
   - Encoder: Takes a layout + room condition → latent vector (mu, logvar)
   - Reparameterization trick: z = mu + sigma * epsilon  (makes sampling differentiable)
   - Decoder: Takes latent z + condition → furniture positions and orientations

2. CNN Evaluator
   - Takes a 2D floorplan image as input
   - Predicts: (a) Is this layout valid? (b) What is its quality score?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONDITIONAL VAE
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Maps [layout x  +  room condition c]  →  mu, logvar

    mu and logvar parameterize the latent Gaussian q(z | x, c).
    """
    def __init__(self, input_dim: int, condition_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim + condition_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            # Second hidden layer
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

    def forward(self, x, condition):
        h = self.net(torch.cat([x, condition], dim=-1))  # concat layout + condition
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    Maps [latent z  +  room condition c]  →  furniture positions + orientations

    Positions: Sigmoid output → always in [0, 1] (inside the room)
    Rotations: Raw logits → Softmax gives probability over 4 facings (N, E, S, W)
    """
    def __init__(self, latent_dim: int, condition_dim: int, num_items: int, hidden: int = 256):
        super().__init__()
        self.num_items = num_items
        self.net = nn.Sequential(
            # First hidden layer
            nn.Linear(latent_dim + condition_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            # Second hidden layer
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        # Position head: (x, y) per item, in [0, 1] via Sigmoid
        self.head_pos = nn.Linear(hidden, num_items * 2)
        # Rotation head: 4 logits per item (N/E/S/W facing)
        self.head_rot = nn.Linear(hidden, num_items * 4)

    def forward(self, z, condition):
        h = self.net(torch.cat([z, condition], dim=-1))
        positions = torch.sigmoid(self.head_pos(h))           # (B, num_items*2)
        rotations = self.head_rot(h).view(-1, self.num_items, 4)  # (B, num_items, 4)
        return positions, rotations


class CVAE(nn.Module):
    """
    Full Conditional VAE = Encoder + Reparameterization + Decoder

    Training:  x, c  →  Encoder  →  z  →  Decoder  →  x_hat
    Inference: sample z ~ N(0, I)  →  Decoder  →  new layout
    """
    def __init__(self, input_dim=24, condition_dim=12, num_items=6, latent_dim=16, hidden=256):
        super().__init__()
        self.encoder = Encoder(input_dim, condition_dim, latent_dim, hidden)
        self.decoder = Decoder(latent_dim, condition_dim, num_items, hidden)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:  z = mu + sigma * epsilon,  epsilon ~ N(0, I)

        WHY: Sampling z ~ N(mu, sigma) is non-differentiable (random node blocks gradients).
        Writing z = mu + sigma * eps moves the randomness into eps, so:
          - dz/d_mu = 1  →  gradient flows to encoder
          - dz/d_sigma = eps  →  gradient flows to encoder
        This is the KEY trick that makes VAE training possible.
        """
        sigma = torch.exp(0.5 * logvar)     # sigma = exp(logvar/2)
        eps   = torch.randn_like(sigma)     # eps ~ N(0, I)
        return mu + sigma * eps             # z = mu + sigma * eps

    def forward(self, x, condition):
        """Full forward pass for TRAINING."""
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        pos, rot = self.decoder(z, condition)
        return pos, rot, mu, logvar

    @torch.no_grad()
    def generate(self, condition, n_samples=1):
        """Generate new layouts at INFERENCE (encoder is not used)."""
        self.eval()
        z = torch.randn(n_samples, self.encoder.fc_mu.out_features)
        pos, rot = self.decoder(z, condition.expand(n_samples, -1))
        return pos, rot


def cvae_loss(pos_pred, pos_true, rot_pred, rot_true, mu, logvar, mask, beta=0.1):
    """
    Total CVAE Loss = Reconstruction Loss + beta * KL Divergence

    Reconstruction Loss:
      - Position:  MAE (Mean Absolute Error)  for (x, y) coordinates
      - Rotation:  Cross-Entropy for the 4-class facing direction

    KL Divergence (closed-form for Gaussian):
      KL( N(mu, sigma^2) || N(0, I) ) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    beta controls how strongly the latent space is regularized toward N(0, I).
    Beta-annealing: start with beta=0 and slowly increase to 0.1.
    """
    B, K = mask.shape

    # --- Position loss (MAE per active item) ---
    pos_pred_2d = pos_pred.view(B, K, 2)
    pos_loss = (F.l1_loss(pos_pred_2d, pos_true, reduction='none') * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)

    # --- Rotation loss (cross-entropy per active item) ---
    rot_loss = (F.cross_entropy(rot_pred.view(B * K, 4), rot_true.view(B * K), reduction='none').view(B, K) * mask).sum() / (mask.sum() + 1e-8)

    # --- KL Divergence (exact closed form) ---
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = pos_loss + rot_loss + beta * kl_loss
    return total, pos_loss, rot_loss, kl_loss


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CNN EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class CNNEvaluator(nn.Module):
    """
    4-layer CNN that scores a 2D floorplan image.

    Input:  (B, 4, 64, 64) image  — 4 channels: room, furniture, fronts, door zone
    Output: validity logit + quality score

    Architecture:
        Conv(4→16) → BN → ReLU → MaxPool(2)   [64 → 32 px]
        Conv(16→32) → BN → ReLU → MaxPool(2)  [32 → 16 px]
        Conv(32→64) → BN → ReLU → MaxPool(2)  [16 → 8 px]
        Conv(64→64) → BN → ReLU → MaxPool(2)  [8 → 4 px]
        Flatten → Dropout → Linear(1024, 128) → ReLU
        head_valid: Linear(128, 1)  → sigmoid = P(valid)
        head_score: Linear(128, 1)  → quality score in [0, 1]
    """
    def __init__(self, in_channels=4, pixels=64):
        super().__init__()

        # 4 convolutional blocks: each halves the spatial size
        self.features = nn.Sequential(
            # Block 1: 64 → 32 px
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            # Block 2: 32 → 16 px
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            # Block 3: 16 → 8 px
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Block 4: 8 → 4 px
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Flatten → (B, 64*4*4) = (B, 1024)
            nn.Flatten(),
        )

        flat_size = 64 * (pixels // 16) ** 2   # = 64 * 4 * 4 = 1024

        self.dense = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
        )
        self.head_valid = nn.Linear(128, 1)   # logit → sigmoid = P(valid)
        self.head_score = nn.Linear(128, 1)   # predicted quality

    def forward(self, image):
        """Returns (validity_logit, quality_score) for a batch of layout images."""
        h = self.dense(self.features(image))
        return self.head_valid(h).squeeze(1), self.head_score(h).squeeze(1)


def evaluator_loss(valid_logit, score_pred, valid_true, score_true, lambda_score=1.0):
    """
    Total Evaluator Loss = BCE(validity) + lambda_score * MSE(quality score)

    BCE (Binary Cross-Entropy) for valid/invalid classification:
      - Uses logit directly: numerically stable (avoids log(sigmoid(x)) overflow)

    MSE for the continuous quality score regression.
    """
    bce  = F.binary_cross_entropy_with_logits(valid_logit, valid_true.float())
    mse  = F.mse_loss(score_pred, score_true.float())
    return bce + lambda_score * mse, bce, mse


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SpaceGen AI — Model Architecture Check")
    print("=" * 50)

    B = 4          # batch size
    K = 6          # number of furniture slots
    INPUT_DIM   = K * 4    # (x, y) + 4-class rotation per item = 4 * 6 = 24
    COND_DIM    = 12       # room width, depth, door info, etc.
    LATENT_DIM  = 16
    PIXELS      = 64

    # --- CVAE ---
    cvae = CVAE(input_dim=INPUT_DIM, condition_dim=COND_DIM,
                num_items=K, latent_dim=LATENT_DIM, hidden=256)
    x_in   = torch.randn(B, INPUT_DIM)
    cond   = torch.randn(B, COND_DIM)
    pos_out, rot_out, mu, logvar = cvae(x_in, cond)
    print(f"\nCVAE Encoder  : ({B}, {INPUT_DIM}+{COND_DIM}) -> mu/logvar shape: {mu.shape}")
    print(f"CVAE Decoder  : ({B}, {LATENT_DIM}+{COND_DIM}) -> positions: {pos_out.shape}, rotations: {rot_out.shape}")

    # CVAE generation (no encoder, z ~ N(0,I))
    gen_pos, gen_rot = cvae.generate(cond[0], n_samples=8)
    print(f"CVAE Generate : 8 new layouts -> positions: {gen_pos.shape}, rotations: {gen_rot.shape}")

    # --- CNN Evaluator ---
    cnn = CNNEvaluator(in_channels=4, pixels=PIXELS)
    images  = torch.randn(B, 4, PIXELS, PIXELS)
    logit, score = cnn(images)
    print(f"\nCNN Evaluator : ({B}, 4, {PIXELS}, {PIXELS}) -> logit: {logit.shape}, score: {score.shape}")
    print(f"\nAll model checks passed!")
