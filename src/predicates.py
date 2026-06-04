"""
LTN Predicates module — Pure PyTorch implementation of LTN-style fuzzy predicates.

All predicates use sigmoid-based fuzzy logic with temperature-controlled
sharpness (tau). The satisfiability of each predicate is a differentiable
scalar in [0, 1] that can be aggregated into an LTN loss.
"""

import torch
import torch.nn as nn
import numpy as np

class P3_Clover_Subset_GDM(nn.Module):
    def forward(self, clover, gdm, tau):
        margin = gdm - clover
        return torch.sigmoid(margin / tau)

class P4_Total_Is_Maximum(nn.Module):
    def forward(self, total, clover, dead, green, gdm, tau):
        others = torch.stack([clover, dead, green, gdm], dim=1)  # [B, 4]
        margins = total.unsqueeze(1) - others
        sats = torch.sigmoid(margins / tau)
        return sats.min(dim=1).values

class P5_NDVI_GDM_Monotonic(nn.Module):
    def forward(self, ndvi_i, ndvi_j, gdm_i, gdm_j, tau):
        agreement = (ndvi_i - ndvi_j) * (gdm_i - gdm_j)
        return torch.sigmoid(agreement / (tau * 10))

class P6_WA_Dead_Zero(nn.Module):
    def forward(self, dead, is_wa_mask, tau):
        if is_wa_mask.sum() == 0:
            return torch.tensor(1.0, device=dead.device)
        wa_dead = dead[is_wa_mask]
        return torch.sigmoid(-wa_dead / tau).mean()

class P7_CloverZeroSpecies(nn.Module):
    def forward(self, clover, is_clover_zero, tau):
        if is_clover_zero.sum() == 0:
            return torch.tensor(1.0, device=clover.device)
        return torch.sigmoid(-clover[is_clover_zero] / tau).mean()

class P8_PureCloverSpecies(nn.Module):
    def forward(self, dead, green, is_pure_clover, tau):
        if is_pure_clover.sum() == 0:
            return torch.tensor(1.0, device=dead.device)
        sat_dead  = torch.sigmoid(-dead[is_pure_clover]  / tau).mean()
        sat_green = torch.sigmoid(-green[is_pure_clover] / tau).mean()
        return torch.min(sat_dead, sat_green)

class P9_WinterDeadProportion(nn.Module):
    def forward(self, dead, total, is_winter, tau):
        if is_winter.sum() == 0:
            return torch.tensor(1.0, device=dead.device)
        prop = dead[is_winter] / total[is_winter].clamp(min=1e-6)
        return torch.sigmoid((prop - 0.24) / tau).mean()

class P10_SummerGreenProportion(nn.Module):
    def forward(self, green, total, is_summer, tau):
        if is_summer.sum() == 0:
            return torch.tensor(1.0, device=green.device)
        prop = green[is_summer] / total[is_summer].clamp(min=1e-6)
        return torch.sigmoid((prop - 0.657) / tau).mean()

class P11_Height_GDM_Monotonic(nn.Module):
    def forward(self, log_height_i, log_height_j, gdm_i, gdm_j, tau):
        agreement = (log_height_i - log_height_j) * (gdm_i - gdm_j)
        return torch.sigmoid(agreement / (tau * 10))


def get_tau_dict(cfg, epoch):
    """
    Computes current tau values based on cosine annealing schedule.
    """
    total_epochs = cfg['training']['epochs']
    tau_end = cfg['ltn']['tau']
    anneal_factor = cfg['ltn']['tau_anneal_factor']
    
    tau_start = {k: v * anneal_factor for k, v in tau_end.items()}
    
    progress = min(epoch / total_epochs, 1.0)
    current_tau = {}
    
    for category in tau_end.keys():
        t = tau_end[category]
        t0 = tau_start[category]
        current_tau[category] = t + 0.5 * (t0 - t) * (1 + np.cos(np.pi * progress))
        
    return current_tau
