"""
Loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def regression_loss(preds, targets_log, target_cols, delta):
    """
    preds: dict of [B] log-space predictions (all 5 targets, including derived)
    targets_log: [B, 5] log-space targets
    delta: Huber transition point, calibrated for log1p space
    """
    total_loss = 0.0
    per_target = {}
    for i, name in enumerate(target_cols):
        y_pred = preds[name]
        y_true = targets_log[:, i]
        loss = F.huber_loss(y_pred, y_true, delta=delta, reduction='mean')
        per_target[name] = loss.item()
        total_loss += loss
    return total_loss, per_target

def gate_loss(clover_presence_logit, y_clover_log, species_is_clover_zero):
    """
    Binary cross-entropy on the presence gate.
    Ground truth: 0 for clover-zero species; (y_clover > 0) for others.
    """
    if clover_presence_logit is None:
        return torch.tensor(0.0, device=y_clover_log.device)
        
    y_present = (y_clover_log > 0).float()                  # [B]
    y_present[species_is_clover_zero] = 0.0                 # Override for known-zero species
    return F.binary_cross_entropy_with_logits(clover_presence_logit, y_present)

def compute_ltn_loss(sat_dict, predicate_weights):
    if not sat_dict:
        return torch.tensor(0.0)
        
    weighted_sats = torch.stack([
        predicate_weights[k] * (1.0 - v) for k, v in sat_dict.items()
    ])
    
    # Normalize weights so they sum to 1
    weight_sum = sum(predicate_weights.values())
    weighted_sats = weighted_sats / weight_sum

    # pMeanError with p=2 over weighted unsatisfied terms
    p = 2
    ltn_loss = (weighted_sats.pow(p).sum()).pow(1.0 / p)
    return ltn_loss


class TotalLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.alpha      = cfg['loss']['alpha']        # Regression loss weight
        self.gamma_gate = cfg['loss']['gamma_gate']   # Clover gate BCE weight
        self.delta      = cfg['loss']['huber_delta']
        self.predicate_weights = cfg['ltn']['predicate_weights']

    def forward(self, preds, targets_log, target_cols, sat_dict,
                clover_logit, y_clover_log, species_is_cz,
                beta=0.5):
        
        L_reg, per_target = regression_loss(preds, targets_log, target_cols, self.delta)
        L_ltn  = compute_ltn_loss(sat_dict, self.predicate_weights)
        L_gate = gate_loss(clover_logit, y_clover_log, species_is_cz)

        total = self.alpha * L_reg + beta * L_ltn + self.gamma_gate * L_gate
        
        breakdown = {
            'L_reg': L_reg.item(), 
            'L_ltn': L_ltn.item() if isinstance(L_ltn, torch.Tensor) else 0.0, 
            'L_gate': L_gate.item() if isinstance(L_gate, torch.Tensor) else 0.0, 
            **per_target
        }
        
        return total, breakdown


def get_ltn_beta(epoch, cfg):
    """LTN loss weight: linear ramp from 0 -> beta_max over warmup epochs.
    Train regression first, then gradually introduce symbolic constraints.
    """
    warmup = cfg['loss']['ltn_beta_warmup']
    beta_max = cfg['loss']['beta_max']
    
    if epoch < warmup:
        return beta_max * (epoch / warmup)
    return beta_max
