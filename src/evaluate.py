"""
Evaluation metrics, constraint satisfaction rate, and drift analysis.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, classification_report
import os
import matplotlib.pyplot as plt
from dataset import TARGET_COLS

def calculate_csr_metrics(preds_orig_dict, epsilon=1.0):
    """
    Calculate exact and approximate Constraint Satisfaction Rates.
    preds_orig_dict: dict mapping 'Dry_Clover_g', etc. to concatenated [N] tensors.
    """
    gdm   = preds_orig_dict['GDM_g']
    total = preds_orig_dict['Dry_Total_g']
    green = preds_orig_dict['Dry_Green_g']
    dead  = preds_orig_dict['Dry_Dead_g']
    clover = preds_orig_dict['Dry_Clover_g']
    
    # CSR-1: GDM = Green + Clover (Should be 1.0)
    csr1 = torch.mean((torch.abs(gdm - (green + clover)) < epsilon).float()).item()
    
    # CSR-2: Total = GDM + Dead (Should be 1.0)
    csr2 = torch.mean((torch.abs(total - (gdm + dead)) < epsilon).float()).item()
    
    # CSR-3: Clover <= GDM
    csr3 = torch.mean((clover <= gdm).float()).item()
    
    # CSR-4: Total >= Max(components)
    max_components = torch.stack([clover, dead, green, gdm], dim=1).max(dim=1).values
    csr4 = torch.mean((total >= max_components).float()).item()
    
    # Symbolic Drift
    drift = torch.mean(torch.abs(total - (green + dead + clover))).item()
    
    return {
        'CSR1_GDM_Conservation': csr1,
        'CSR2_Total_Conservation': csr2,
        'CSR3_Clover_LTE_GDM': csr3,
        'CSR4_Total_GTE_Max': csr4,
        'Symbolic_Drift_g': drift
    }


def full_evaluation_report(val_loader, all_preds_orig, all_targets_orig, model, device, output_dir):
    """
    Comprehensive evaluation and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {}
    
    # --- 1. Regression Metrics (Original Space) ---
    for col in TARGET_COLS:
        y_true = torch.cat(all_targets_orig[col]).numpy()
        y_pred = torch.cat(all_preds_orig[col]).numpy()
        
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (exclude true=0)
        mask = y_true > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
            
        metrics[col] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
    # --- 2. CSR Metrics ---
    # Convert lists of tensors to single tensors
    preds_orig_tensor = {k: torch.cat(v) for k, v in all_preds_orig.items()}
    csr_metrics = calculate_csr_metrics(preds_orig_tensor)
    metrics['CSR'] = csr_metrics
    
    # --- 3. Save Summary ---
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write("=== Biomass Prediction Evaluation ===\n\n")
        f.write("--- Regression Metrics (Original Space) ---\n")
        for col in TARGET_COLS:
            f.write(f"\n{col}:\n")
            for k, v in metrics[col].items():
                f.write(f"  {k}: {v:.4f}\n")
                
        f.write("\n--- Constraint Satisfaction Rates ---\n")
        for k, v in csr_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    return metrics
