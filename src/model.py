"""
Full BiomassLTNModel Assembly.
"""

import torch
import torch.nn as nn
from encoders import DualImageEncoder, TabularEncoder
from fusion import CrossModalAttentionFusion
from heads import RegressionHead, CloverHead
from symbolic import SymbolicConservationLayer
from predicates import (
    P3_Clover_Subset_GDM, P4_Total_Is_Maximum, P5_NDVI_GDM_Monotonic,
    P6_WA_Dead_Zero, P7_CloverZeroSpecies, P8_PureCloverSpecies,
    P9_WinterDeadProportion, P10_SummerGreenProportion, P11_Height_GDM_Monotonic
)

class BiomassLTNModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Encoders
        enc_cfg = cfg['model']['img_encoder']
        self.img_encoder = DualImageEncoder(
            out_dim=enc_cfg['out_dim'],
            eff_dropout=enc_cfg['eff_dropout'],
            freeze_eff=enc_cfg['freeze_eff_blocks'],
            freeze_vit_pct=enc_cfg['freeze_vit_pct']
        )
        self.tab_encoder = TabularEncoder(cfg)
        
        # Fusion
        fus_cfg = cfg['model']['fusion']
        self.fusion = CrossModalAttentionFusion(
            img_dim=fus_cfg['img_dim'],
            tab_dim=fus_cfg['tab_dim'],
            attn_dim=fus_cfg['attn_dim'],
            hidden_dim=fus_cfg['hidden_dim'],
            dropout=fus_cfg['dropout']
        )
        
        # Heads
        head_cfg = cfg['model']['heads']
        hidden = head_cfg['hidden_dims'][0]
        dropout = head_cfg['dropout']
        
        # Z dimension is hidden_dim from fusion (256)
        self.green_head  = RegressionHead(256, hidden, dropout)
        self.dead_head   = RegressionHead(256, hidden, dropout)
        
        if head_cfg['use_clover_gate']:
            self.clover_head = CloverHead(256, dropout)
        else:
            self.clover_head = RegressionHead(256, hidden, dropout)
            
        self.symbolic = SymbolicConservationLayer()

        # LTN Predicates (P3-P11)
        self.P3  = P3_Clover_Subset_GDM()
        self.P4  = P4_Total_Is_Maximum()
        self.P5  = P5_NDVI_GDM_Monotonic()
        self.P6  = P6_WA_Dead_Zero()
        self.P7  = P7_CloverZeroSpecies()
        self.P8  = P8_PureCloverSpecies()
        self.P9  = P9_WinterDeadProportion()
        self.P10 = P10_SummerGreenProportion()
        self.P11 = P11_Height_GDM_Monotonic()

    def forward(self, images, tab_batch, tau_dict):
        # Encode
        f_vis = self.img_encoder(images)
        f_tab = self.tab_encoder(tab_batch)
        Z     = self.fusion(f_vis, f_tab)

        # Primary predictions (log1p space)
        y_hat_green_log  = self.green_head(Z)
        y_hat_dead_log   = self.dead_head(Z)
        y_hat_clover_log = self.clover_head(Z)
        
        if isinstance(self.clover_head, CloverHead):
            clover_logit = self.clover_head.presence_logit(Z)
        else:
            clover_logit = None

        # Symbolic derivation (original space and log space combined)
        derived = self.symbolic(y_hat_green_log, y_hat_dead_log, y_hat_clover_log)
        
        # derived contains all 5 log-space preds + 5 original-space preds (_green etc.)

        # LTN satisfiabilities
        orig = {k.lstrip('_'): derived[k] for k in derived if k.startswith('_')}
        sat_dict = self._compute_satisfiabilities(orig, tab_batch, tau_dict)

        return derived, sat_dict, clover_logit

    def _compute_satisfiabilities(self, orig, tab, tau_dict):
        ndvi = tab['numerical'][:, 0]   # scaled NDVI
        h    = tab['numerical'][:, 1]   # scaled log1p(Height)

        # Pairwise predicates: use random half-batch pairs for efficiency
        B = orig['green'].size(0)
        idx_i = torch.randperm(B)[:B//2]
        idx_j = torch.randperm(B)[:B//2]

        return {
            'P3':  self.P3(orig['clover'], orig['gdm'], tau_dict['hard']).mean(),
            'P4':  self.P4(orig['total'], orig['clover'],
                           orig['dead'], orig['green'], orig['gdm'], tau_dict['hard']).mean(),
            'P5':  self.P5(ndvi[idx_i], ndvi[idx_j],
                           orig['gdm'][idx_i], orig['gdm'][idx_j], tau_dict['very_soft']).mean(),
            'P6':  self.P6(orig['dead'], tab['is_wa'], tau_dict['near_exact']),
            'P7':  self.P7(orig['clover'], tab['is_clover_zero'], tau_dict['near_exact']),
            'P8':  self.P8(orig['dead'], orig['green'], tab['is_pure_clover'], tau_dict['near_exact']),
            'P9':  self.P9(orig['dead'], orig['total'], tab['is_winter'], tau_dict['soft']),
            'P10': self.P10(orig['green'], orig['total'], tab['is_summer'], tau_dict['soft']),
            'P11': self.P11(h[idx_i], h[idx_j],
                            orig['gdm'][idx_i], orig['gdm'][idx_j], tau_dict['very_soft']).mean(),
        }
