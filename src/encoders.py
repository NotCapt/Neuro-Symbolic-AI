"""
Encoders module — Image and Tabular feature extractors.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class DualImageEncoder(nn.Module):
    def __init__(self, out_dim=256, eff_dropout=0.3, freeze_eff=4, freeze_vit_pct=0.75):
        super().__init__()

        # --- EfficientNet-B0: local texture, vegetation density, colour ---
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        features = list(backbone.features.children())
        for i, block in enumerate(features):
            if i < freeze_eff:
                for p in block.parameters():
                    p.requires_grad = False
        self.eff_backbone = backbone.features
        self.eff_pool     = nn.AdaptiveAvgPool2d(1)
        self.eff_proj     = nn.Sequential(
            nn.Dropout(eff_dropout),
            nn.Linear(1280, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

        # --- ViT-B/16: global semantics, pasture composition ---
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        n_blocks = len(vit.encoder.layers)
        freeze_n = int(n_blocks * freeze_vit_pct)  # freeze ~75% of transformer blocks
        for i, block in enumerate(vit.encoder.layers):
            if i < freeze_n:
                for p in block.parameters():
                    p.requires_grad = False
        # Store ViT components separately so we can extract CLS token
        # (the full vit.forward() returns 1000-dim logits, not 768-dim features)
        self.vit_conv_proj   = vit.conv_proj
        self.vit_class_token = vit.class_token
        self.vit_pos_embed   = vit.encoder.pos_embedding
        self.vit_encoder_ln  = vit.encoder.ln
        self.vit_encoder_layers = vit.encoder.layers
        self.vit_seq_length  = vit.seq_length
        self.vit_hidden_dim  = vit.hidden_dim

        self.vit_proj = nn.Sequential(
            nn.Linear(768, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )

    def _vit_features(self, x):
        """Extract CLS token (768-dim) from ViT, bypassing the classification head."""
        # Patch embedding
        B = x.shape[0]
        x = self.vit_conv_proj(x)             # [B, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)      # [B, n_patches, hidden_dim]

        # Prepend CLS token
        cls_tokens = self.vit_class_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) # [B, 1+n_patches, hidden_dim]

        # Add positional embedding
        x = x + self.vit_pos_embed

        # Transformer encoder blocks
        for layer in self.vit_encoder_layers:
            x = layer(x)

        x = self.vit_encoder_ln(x)
        return x[:, 0]                         # [B, 768] — CLS token only

    def forward(self, x):
        # EfficientNet branch
        f_eff = self.eff_backbone(x)            # [B, 1280, 7, 7]
        f_eff = self.eff_pool(f_eff).flatten(1) # [B, 1280]
        f_eff = self.eff_proj(f_eff)            # [B, 256]

        # ViT branch (CLS token)
        f_vit = self._vit_features(x)          # [B, 768]
        f_vit = self.vit_proj(f_vit)            # [B, 256]

        return torch.cat([f_eff, f_vit], dim=1)  # [B, 512]


class TabularEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        tab_cfg = cfg['model']['tabular_encoder']
        
        # Embeddings
        self.species_emb = nn.Embedding(15, tab_cfg['species_emb_dim']) 
        self.state_emb   = nn.Embedding(4, tab_cfg['state_emb_dim'])
        self.month_emb   = nn.Embedding(12, tab_cfg['month_emb_dim'])
        self.season_emb  = nn.Embedding(4, tab_cfg['season_emb_dim'])
        
        # Calculate total dimension
        cat_dim = (tab_cfg['species_emb_dim'] + 
                   tab_cfg['state_emb_dim'] + 
                   tab_cfg['month_emb_dim'] + 
                   tab_cfg['season_emb_dim'])
        num_dim = 2 # NDVI, Height
        in_dim = cat_dim + num_dim
        
        hidden_dim = tab_cfg['hidden_dim']
        dropout = tab_cfg['dropout']

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, tab):
        cat = torch.cat([
            self.species_emb(tab['species']),
            self.state_emb(tab['state']),
            self.month_emb(tab['month']),
            self.season_emb(tab['season']),
            tab['numerical']                      # [NDVI, log1p_Height], normalized
        ], dim=1)
        return self.mlp(cat)   # [B, 128]
