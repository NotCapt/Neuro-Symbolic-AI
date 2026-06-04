"""
Cross-Modal Attention Fusion Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, img_dim=512, tab_dim=128, attn_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        # Cross-attention: image (visual) queries tabular context
        self.q = nn.Linear(img_dim, attn_dim)
        self.k = nn.Linear(tab_dim, attn_dim)
        self.v = nn.Linear(tab_dim, attn_dim)
        self.scale = attn_dim ** -0.5

        # FiLM-style conditioning: tabular generates γ, β for visual features
        self.film_gamma = nn.Linear(tab_dim, img_dim)
        self.film_beta  = nn.Linear(tab_dim, img_dim)

        # Fusion MLP
        fused_dim = img_dim + tab_dim + attn_dim  # 512+128+128 = 768
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, f_img, f_tab):
        # FiLM modulation: metadata conditions the visual features
        gamma = self.film_gamma(f_tab)                      # [B, 512]
        beta  = self.film_beta(f_tab)                       # [B, 512]
        f_img_cond = gamma * f_img + beta                   # [B, 512]

        # Cross-modal attention
        Q = self.q(f_img_cond).unsqueeze(1)                 # [B, 1, 128]
        K = self.k(f_tab).unsqueeze(1)                      # [B, 1, 128]
        V = self.v(f_tab).unsqueeze(1)                      # [B, 1, 128]
        attn_w = F.softmax(Q @ K.transpose(-2,-1) * self.scale, dim=-1)
        f_attn = (attn_w @ V).squeeze(1)                    # [B, 128]

        # Fuse all representations
        fused = torch.cat([f_img_cond, f_tab, f_attn], dim=1)  # [B, 768]
        return self.mlp(fused)                               # [B, 256] = Z
