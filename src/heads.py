"""
Regression heads module.
"""

import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    def __init__(self, in_dim=256, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()    # Output >= 0 in log1p space (log1p(0) = 0)
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)   # [B]


class CloverHead(nn.Module):
    """Two-stage: (1) binary presence gate, (2) conditional amount regression."""
    def __init__(self, in_dim=256, dropout=0.2):
        super().__init__()
        # Stage 1: Is clover present?
        self.presence_gate = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)        # logit: sigmoid(.) = P(clover > 0)
        )
        # Stage 2: How much clover? (predicted only when present)
        self.amount_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, z):
        p_present = torch.sigmoid(self.presence_gate(z))    # [B, 1] in (0,1)
        amount    = self.amount_net(z)                       # [B, 1] > 0
        return (p_present * amount).squeeze(-1)              # [B], gated prediction

    def presence_logit(self, z):
        return self.presence_gate(z).squeeze(-1)             # [B], for BCE loss


class HeteroscedasticHead(nn.Module):
    def __init__(self, in_dim=256, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.mu_head    = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.logvar_head = nn.Linear(hidden, 1)   # unconstrained

    def forward(self, z):
        h = self.shared(z)
        mu     = self.mu_head(h).squeeze(-1)       # [B]
        logvar = self.logvar_head(h).squeeze(-1)   # [B]
        return mu, logvar
