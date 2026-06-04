"""
Main training and evaluation entry point.
"""

import os
import sys

# Ensure src/ is on the path when run as `python src/main.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
import torch
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from dataset import load_and_preprocess, split_data, fit_scaler, build_dataloaders
from model import BiomassLTNModel
from loss import TotalLoss
from train import train_epoch, val_epoch
from evaluate import full_evaluation_report

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Set up device and seeds
    set_seed(cfg['data'].get('random_seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    outputs_dir = os.path.join(base_dir, 'outputs')
    checkpoints_dir = os.path.join(outputs_dir, 'checkpoints')
    metrics_dir = os.path.join(outputs_dir, 'metrics')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # 1. Data Pipeline
    print("Loading and preprocessing data...")
    csv_path = os.path.join(base_dir, cfg['data']['train_csv'])
    img_root = os.path.join(base_dir, cfg['data']['img_root'])
    
    df_wide, label_encoders = load_and_preprocess(csv_path, img_root, cfg)
    train_df, val_df = split_data(df_wide, cfg)
    train_df, val_df, scaler = fit_scaler(train_df, val_df, outputs_dir)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Data loaders will be rebuilt per epoch if curriculum learning is active
    train_loader, val_loader = build_dataloaders(train_df, val_df, cfg, epoch=0)

    # 2. Model setup
    print("Building model...")
    model = BiomassLTNModel(cfg).to(device)
    
    loss_fn = TotalLoss(cfg).to(device)

    # 3. Optimizer & Scheduler
    # Differential LR for backbone vs other parts
    backbone_params = list(model.img_encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('img_encoder.')]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg['training']['optimizer']['lr_backbone'], 
         'weight_decay': cfg['training']['optimizer']['weight_decay_backbone']},
        {'params': other_params, 'lr': cfg['training']['optimizer']['lr_other'], 
         'weight_decay': cfg['training']['optimizer']['weight_decay_other']},
    ])

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg['training']['scheduler']['warmup_epochs'])
    cosine = CosineAnnealingLR(optimizer, T_max=cfg['training']['scheduler']['T_max'] - cfg['training']['scheduler']['warmup_epochs'], 
                               eta_min=cfg['training']['scheduler']['eta_min'])
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg['training']['scheduler']['warmup_epochs']])

    # AMP Scaler
    use_amp = cfg['training'].get('amp', False) and torch.cuda.is_available()
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 4. Training Loop
    epochs = cfg['training']['epochs']
    patience = cfg['training']['patience']
    best_val_rmse = float('inf')
    epochs_without_improve = 0

    print("Starting training...")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Rebuild train loader if curriculum is active and in warmup phase
        curriculum_cfg = cfg['training'].get('curriculum', {})
        if curriculum_cfg.get('enabled', True) and epoch < curriculum_cfg.get('warmup_epochs', 30):
            train_loader, _ = build_dataloaders(train_df, val_df, cfg, epoch=epoch)

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, amp_scaler, epoch, device, cfg)
        val_metrics, all_preds, all_targets = val_epoch(model, val_loader, loss_fn, epoch, device, cfg)
        
        scheduler.step()

        # Print some key metrics
        train_loss = train_metrics['L_reg'] + train_metrics.get('L_ltn', 0)
        val_loss = val_metrics['L_reg'] + val_metrics.get('L_ltn', 0)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Track mean primary target RMSE
        primary_targets = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
        mean_val_rmse = np.mean([val_metrics[f'RMSE_orig_{t}'] for t in primary_targets])
        print(f"Mean Val RMSE (Original Space): {mean_val_rmse:.4f}")

        # Checkpoint logic
        if mean_val_rmse < best_val_rmse:
            best_val_rmse = mean_val_rmse
            epochs_without_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': cfg
            }, os.path.join(checkpoints_dir, 'best_model.pt'))
            
            # Save final evaluation report for the best model
            full_evaluation_report(val_loader, all_preds, all_targets, model, device, metrics_dir)
            print(">> Best model saved!")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # Save last model
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'config': cfg
    }, os.path.join(checkpoints_dir, 'last_model.pt'))
    
    print("\nTraining completed.")

if __name__ == '__main__':
    main()
