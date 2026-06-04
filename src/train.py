"""
Training loop and utilities.
"""

import torch
from collections import defaultdict
import numpy as np
from loss import get_ltn_beta
from predicates import get_tau_dict
from dataset import TARGET_COLS

def train_epoch(model, loader, optimizer, loss_fn, scaler, epoch, device, cfg):
    model.train()
    running = defaultdict(float)
    beta = get_ltn_beta(epoch, cfg)
    tau_dict = get_tau_dict(cfg, epoch)
    use_amp = cfg['training'].get('amp', False)

    for images, tab_d, targets in loader:
        images  = images.to(device)
        targets = targets.to(device)     # [B, 5] in log1p space
        tab_d   = {k: v.to(device) for k, v in tab_d.items()}

        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                preds, sat_dict, clover_logit = model(images, tab_d, tau_dict)
                loss, breakdown = loss_fn(
                    preds, targets, TARGET_COLS, sat_dict,
                    clover_logit, targets[:, TARGET_COLS.index('Dry_Clover_g')],
                    tab_d['is_clover_zero'], beta=beta
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            preds, sat_dict, clover_logit = model(images, tab_d, tau_dict)
            loss, breakdown = loss_fn(
                preds, targets, TARGET_COLS, sat_dict,
                clover_logit, targets[:, TARGET_COLS.index('Dry_Clover_g')],
                tab_d['is_clover_zero'], beta=beta
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['training']['grad_clip'])
            optimizer.step()

        for k, v in breakdown.items():
            running[k] += v

    return {k: v / len(loader) for k, v in running.items()}


def val_epoch(model, loader, loss_fn, epoch, device, cfg):
    model.eval()
    running = defaultdict(float)
    beta = get_ltn_beta(epoch, cfg)
    tau_dict = get_tau_dict(cfg, epoch)
    
    # Storage for evaluation metrics
    all_preds_orig = defaultdict(list)
    all_targets_orig = defaultdict(list)
    all_sat_dict = defaultdict(list)
    
    with torch.no_grad():
        for images, tab_d, targets in loader:
            images  = images.to(device)
            targets = targets.to(device)
            tab_d   = {k: v.to(device) for k, v in tab_d.items()}

            preds, sat_dict, clover_logit = model(images, tab_d, tau_dict)
            
            # Loss computation
            loss, breakdown = loss_fn(
                preds, targets, TARGET_COLS, sat_dict,
                clover_logit, targets[:, TARGET_COLS.index('Dry_Clover_g')],
                tab_d['is_clover_zero'], beta=beta
            )
            
            for k, v in breakdown.items():
                running[k] += v
                
            # Collect original-space predictions for metrics
            # Note: targets are in log1p space, need expm1
            # Map target column names to internal orig-space keys from SymbolicConservationLayer
            _COL_TO_ORIG = {
                'Dry_Clover_g': '_clover',
                'Dry_Dead_g':   '_dead',
                'Dry_Green_g':  '_green',
                'Dry_Total_g':  '_total',
                'GDM_g':        '_gdm',
            }
            for i, col in enumerate(TARGET_COLS):
                # Back-transform targets
                t_orig = torch.expm1(targets[:, i].clamp(min=0))
                all_targets_orig[col].append(t_orig.cpu())
                
                # Predictions (already available in orig space from symbolic layer)
                orig_key = _COL_TO_ORIG[col]
                all_preds_orig[col].append(preds[orig_key].cpu())
                
            # Collect satisfiabilities
            for k, v in sat_dict.items():
                # Some are scalars, some might be batch-wise. Average scalars.
                if v.dim() == 0:
                    all_sat_dict[k].append(v.item())
                else:
                    all_sat_dict[k].extend(v.cpu().tolist())

    metrics = {k: v / len(loader) for k, v in running.items()}
    
    # Compute RMSE in original space
    for col in TARGET_COLS:
        y_true = torch.cat(all_targets_orig[col])
        y_pred = torch.cat(all_preds_orig[col])
        rmse = torch.sqrt(torch.mean((y_pred - y_true)**2)).item()
        metrics[f'RMSE_orig_{col}'] = rmse
        
    # Average satisfiabilities
    for k in all_sat_dict:
        metrics[f'SAT_{k}'] = np.mean(all_sat_dict[k])
        
    return metrics, all_preds_orig, all_targets_orig
