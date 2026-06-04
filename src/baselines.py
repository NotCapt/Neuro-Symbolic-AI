"""
Baseline models for comparison.
"""

import torch
import torch.nn as nn
from encoders import DualImageEncoder, TabularEncoder
from heads import RegressionHead, CloverHead
from symbolic import SymbolicConservationLayer
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from dataset import CLOVER_ZERO_SPECIES

# --- B1: XGBoost (Tabular Only) ---
def build_xgboost_baseline(random_state=42):
    """
    Builds the XGBoost tabular baseline.
    Wrapped in MultiOutputRegressor to predict all 5 targets.
    """
    base_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        objective='reg:squarederror'
    )
    return MultiOutputRegressor(base_model)

def apply_xgboost_rules(preds, states, species, target_cols):
    """
    Applies strict structural rules to XGBoost predictions as a post-processing step.
    preds: numpy array of predictions [B, 5]
    """
    import numpy as np
    
    dead_idx = target_cols.index('Dry_Dead_g')
    clover_idx = target_cols.index('Dry_Clover_g')
    green_idx = target_cols.index('Dry_Green_g')
    total_idx = target_cols.index('Dry_Total_g')
    
    # 1. State=WA -> Dead=0
    preds[states == 'WA', dead_idx] = 0.0
    
    # 2. Clover-zero species -> Clover=0
    for cz in CLOVER_ZERO_SPECIES:
        mask = [s == cz or s.startswith(cz + '_') for s in species]
        preds[mask, clover_idx] = 0.0
        
    # 3. Additive constraint: Total = Green + Dead + Clover
    # (Since this is original space, just sum them up)
    preds[:, total_idx] = preds[:, green_idx] + preds[:, dead_idx] + preds[:, clover_idx]
    
    return preds


# --- B2: Tabular-Only Neural ---
class TabularOnlyBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tab_encoder = TabularEncoder(cfg)
        
        # 5 independent heads (no conservation layer)
        hidden = cfg['model']['heads']['hidden_dims'][0]
        dropout = cfg['model']['heads']['dropout']
        
        self.green_head  = RegressionHead(128, hidden, dropout)
        self.dead_head   = RegressionHead(128, hidden, dropout)
        self.clover_head = RegressionHead(128, hidden, dropout)
        self.gdm_head    = RegressionHead(128, hidden, dropout)
        self.total_head  = RegressionHead(128, hidden, dropout)

    def forward(self, tab_batch):
        f_tab = self.tab_encoder(tab_batch)
        
        return {
            'Dry_Green_g':  self.green_head(f_tab),
            'Dry_Dead_g':   self.dead_head(f_tab),
            'Dry_Clover_g': self.clover_head(f_tab),
            'GDM_g':        self.gdm_head(f_tab),
            'Dry_Total_g':  self.total_head(f_tab)
        }, {}, None


# --- B3: Image-Only Neural ---
class ImageOnlyBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        enc_cfg = cfg['model']['img_encoder']
        self.img_encoder = DualImageEncoder(
            out_dim=enc_cfg['out_dim'],
            eff_dropout=enc_cfg['eff_dropout'],
            freeze_eff=enc_cfg['freeze_eff_blocks'],
            freeze_vit_pct=enc_cfg['freeze_vit_pct']
        )
        
        # 5 independent heads from 512-dim visual features
        hidden = cfg['model']['heads']['hidden_dims'][0]
        dropout = cfg['model']['heads']['dropout']
        
        self.green_head  = RegressionHead(512, hidden, dropout)
        self.dead_head   = RegressionHead(512, hidden, dropout)
        self.clover_head = RegressionHead(512, hidden, dropout)
        self.gdm_head    = RegressionHead(512, hidden, dropout)
        self.total_head  = RegressionHead(512, hidden, dropout)

    def forward(self, images):
        f_vis = self.img_encoder(images)
        
        return {
            'Dry_Green_g':  self.green_head(f_vis),
            'Dry_Dead_g':   self.dead_head(f_vis),
            'Dry_Clover_g': self.clover_head(f_vis),
            'GDM_g':        self.gdm_head(f_vis),
            'Dry_Total_g':  self.total_head(f_vis)
        }, {}, None


# --- B4: Full Neural (No LTN, No Conservation) ---
class FullNeuralNoConservation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from model import BiomassLTNModel
        # Re-use components from main model to ensure exact same architecture
        self.core = BiomassLTNModel(cfg)
        
        # Replace symbolic layer with independent heads for GDM and Total
        hidden = cfg['model']['heads']['hidden_dims'][0]
        dropout = cfg['model']['heads']['dropout']
        
        self.gdm_head   = RegressionHead(256, hidden, dropout)
        self.total_head = RegressionHead(256, hidden, dropout)

    def forward(self, images, tab_batch, tau_dict=None):
        f_vis = self.core.img_encoder(images)
        f_tab = self.core.tab_encoder(tab_batch)
        Z     = self.core.fusion(f_vis, f_tab)

        y_green  = self.core.green_head(Z)
        y_dead   = self.core.dead_head(Z)
        y_clover = self.core.clover_head(Z)
        y_gdm    = self.gdm_head(Z)
        y_total  = self.total_head(Z)

        clover_logit = None
        if hasattr(self.core.clover_head, 'presence_logit'):
            clover_logit = self.core.clover_head.presence_logit(Z)

        # Back-transform to original space manually for evaluation (no symbolic layer used)
        orig = {
            'green':  torch.expm1(y_green.clamp(min=0)),
            'dead':   torch.expm1(y_dead.clamp(min=0)),
            'clover': torch.expm1(y_clover.clamp(min=0)),
            'gdm':    torch.expm1(y_gdm.clamp(min=0)),
            'total':  torch.expm1(y_total.clamp(min=0))
        }

        preds = {
            'Dry_Green_g':  y_green,
            'Dry_Dead_g':   y_dead,
            'Dry_Clover_g': y_clover,
            'GDM_g':        y_gdm,
            'Dry_Total_g':  y_total,
            '_green':  orig['green'],
            '_dead':   orig['dead'],
            '_clover': orig['clover'],
            '_gdm':    orig['gdm'],
            '_total':  orig['total'],
        }

        return preds, {}, clover_logit


# --- B5: Full Neural (With Conservation, No LTN) ---
class FullNeuralWithConservation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from model import BiomassLTNModel
        self.core = BiomassLTNModel(cfg)

    def forward(self, images, tab_batch, tau_dict=None):
        # Forward pass exactly as main model, but return empty satisfiabilities dict
        # effectively disabling the LTN loss component.
        derived, _, clover_logit = self.core(images, tab_batch, tau_dict)
        return derived, {}, clover_logit
