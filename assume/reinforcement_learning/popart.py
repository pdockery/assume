"""PopArt: Preserving Outputs Precisely, while Adaptively Rescaling Targets.

Adaptive normalization layer for critic networks that maintains running
mean/std of targets and corrects weights to preserve output invariance.

Reference: van Hasselt et al. (2016) "Learning values across many orders of magnitude"
"""

import torch
import torch.nn as nn


class PopArtLayer(nn.Module):
    """Drop-in replacement for a final nn.Linear layer with adaptive normalization.

    Maintains running EMA estimates of target mean (mu) and second moment (nu).
    The forward pass is identical to nn.Linear. Normalization and denormalization
    are explicit operations called during training.

    Args:
        in_features: Input dimension.
        out_features: Output dimension (typically 1 for Q-value).
        beta: EMA decay rate for statistics updates.
    """

    def __init__(self, in_features: int, out_features: int = 1, beta: float = 3e-4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.beta = beta

        # Raw (uncorrected) EMA accumulators
        self.register_buffer("_mu_raw", torch.zeros(out_features))
        self.register_buffer("_nu_raw", torch.ones(out_features))

        # Bias-corrected statistics used for normalize/denormalize
        self.register_buffer("mu", torch.zeros(out_features))
        self.register_buffer("sigma", torch.ones(out_features))
        self.register_buffer("n_updates", torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard linear forward — identical to nn.Linear."""
        return self.linear(x)

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize raw targets: (y - mu) / sigma."""
        return (targets - self.mu) / self.sigma.clamp(min=1e-6)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized critic output back to real scale: out * sigma + mu."""
        return normalized * self.sigma + self.mu

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        """Update running mean/std from a batch of raw targets, then correct weights.

        This must be called BEFORE computing the normalized loss for the current batch.
        The weight correction ensures that the output of the layer is unchanged
        despite the statistics update.

        Args:
            targets: Raw (unnormalized) target values, shape [batch, out_features] or [batch].
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Save old statistics for weight correction
        mu_old = self.mu.clone()
        sigma_old = self.sigma.clone()

        # EMA update on raw (uncorrected) accumulators
        self.n_updates += 1
        batch_mean = targets.mean(dim=0)
        batch_second = (targets**2).mean(dim=0)

        self._mu_raw.mul_(1 - self.beta).add_(batch_mean * self.beta)
        self._nu_raw.mul_(1 - self.beta).add_(batch_second * self.beta)

        # Bias correction
        bias_corr = 1 - (1 - self.beta) ** self.n_updates.float()
        mu_corrected = self._mu_raw / bias_corr
        nu_corrected = self._nu_raw / bias_corr

        # Update public stats
        self.mu.copy_(mu_corrected)
        self.sigma.copy_(
            (nu_corrected - mu_corrected**2).clamp(min=1e-6).sqrt()
        )

        # Output-preserving weight correction:
        #   w_new = (sigma_old / sigma_new) * w_old
        #   b_new = (sigma_old * b_old + mu_old - mu_new) / sigma_new
        sigma_new = self.sigma.clamp(min=1e-6)
        sigma_old_safe = sigma_old.clamp(min=1e-6)
        ratio = sigma_old_safe / sigma_new

        self.linear.weight.data.mul_(ratio.unsqueeze(1))
        self.linear.bias.data.mul_(ratio).add_(
            (mu_old - self.mu) / sigma_new
        )
