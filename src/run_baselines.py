"""
Baseline Training, Evaluation, and Comparative Analysis.

Implements the full comparative study required by the project specification:
  B1: XGBoost (Tabular Only) - symbolic/rule-based post-processing
  B2: Neural Tabular Only - metadata encoder + independent heads
  B3: Neural Image Only - dual image encoder + independent heads
  B4: Full Neural (No Conservation, No LTN) - multi-modal, no constraints
  B5: Full Neural (With Conservation, No LTN) - multi-modal, conservation only
  Main: BiomassLTNModel - full neuro-symbolic approach (loaded from checkpoint)
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (load_and_preprocess, split_data, fit_scaler,
                     build_dataloaders, TARGET_COLS)
from loss import regression_loss
from predicates import get_tau_dict
from evaluate import calculate_csr_metrics
from baselines import (build_xgboost_baseline, apply_xgboost_rules,
                       TabularOnlyBaseline, ImageOnlyBaseline,
                       FullNeuralNoConservation, FullNeuralWithConservation)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_regression_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2, MAPE in original space (numpy arrays)."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mask = y_true > 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float('nan')
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


def _preds_to_orig_space(preds_dict):
    """
    Convert log-space predictions to original space.
    Handles both models that provide _green/_dead/... keys (original space)
    and those that only provide log-space target-name keys.
    """
    _COL_TO_ORIG = {
        'Dry_Clover_g': '_clover',
        'Dry_Dead_g':   '_dead',
        'Dry_Green_g':  '_green',
        'Dry_Total_g':  '_total',
        'GDM_g':        '_gdm',
    }
    result = {}
    for col, orig_key in _COL_TO_ORIG.items():
        if orig_key in preds_dict:
            result[col] = preds_dict[orig_key]
        elif col in preds_dict:
            result[col] = torch.expm1(preds_dict[col].clamp(min=0))
    return result


def _print_metrics(name, metrics):
    """Pretty-print metrics for a single model."""
    print(f"\n  Results for {name}:")
    for col in TARGET_COLS:
        if col in metrics:
            m = metrics[col]
            print(f"    {col}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}"
                  f"  R2={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%")
    if 'CSR' in metrics:
        csr = metrics['CSR']
        print("    CSR: ", end="")
        for k, v in csr.items():
            print(f"{k}={v:.4f} ", end="")
        print()


# ─── B1: XGBoost ─────────────────────────────────────────────────────────────

def train_and_evaluate_xgboost(train_df, val_df, cfg):
    """Train and evaluate B1: XGBoost (Tabular Only) baseline."""
    print("\n" + "=" * 60)
    print("B1: XGBoost (Tabular Only)")
    print("=" * 60)

    feature_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm',
                    'Species_idx', 'State_idx', 'month_idx', 'season_idx']

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COLS].values          # log1p space
    X_val   = val_df[feature_cols].values
    y_val   = val_df[TARGET_COLS].values             # log1p space

    model = build_xgboost_baseline(
        random_state=cfg['data'].get('random_seed', 42))
    print("  Training XGBoost ...")
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_val)

    # Back-transform to original space
    y_pred_orig = np.expm1(np.clip(y_pred_log, 0, None))
    y_true_orig = np.expm1(np.clip(y_val, 0, None))

    # Apply hard symbolic rules
    y_pred_orig = apply_xgboost_rules(
        y_pred_orig,
        val_df['State'].values,
        val_df['Species'].values,
        TARGET_COLS,
    )

    # Regression metrics
    metrics = {}
    for i, col in enumerate(TARGET_COLS):
        metrics[col] = _compute_regression_metrics(
            y_true_orig[:, i], y_pred_orig[:, i])

    # CSR metrics
    preds_t = {col: torch.tensor(y_pred_orig[:, i], dtype=torch.float32)
               for i, col in enumerate(TARGET_COLS)}
    metrics['CSR'] = calculate_csr_metrics(preds_t)

    _print_metrics("B1: XGBoost", metrics)
    return metrics


# ─── Generic Neural Baseline Training ────────────────────────────────────────

def _train_neural_baseline(model, model_type, train_loader, val_loader,
                           cfg, device, baseline_epochs=None):
    """
    Train a neural baseline and return evaluation metrics.

    model_type : 'tabular_only' | 'image_only' | 'full'
    """
    epochs = baseline_epochs or cfg['training']['epochs']
    delta  = cfg['loss']['huber_delta']

    if device.type == 'cpu':
        print("  [CPU Optimization] Freezing image encoder backbone to accelerate training...")
        img_encoder = None
        if hasattr(model, 'img_encoder'):
            img_encoder = model.img_encoder
        elif hasattr(model, 'core') and hasattr(model.core, 'img_encoder'):
            img_encoder = model.core.img_encoder

        if img_encoder is not None:
            # Freeze EfficientNet features
            for p in img_encoder.eff_backbone.parameters():
                p.requires_grad = False
            # Freeze ViT layers
            for p in img_encoder.vit_encoder_layers.parameters():
                p.requires_grad = False
            if hasattr(img_encoder, 'vit_conv_proj'):
                for p in img_encoder.vit_conv_proj.parameters():
                    p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['training']['optimizer']['lr_other'],
        weight_decay=cfg['training']['optimizer']['weight_decay_other'],
    )

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # ── train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for images, tab_d, targets in train_loader:
            images  = images.to(device)
            targets = targets.to(device)
            tab_d   = {k: v.to(device) for k, v in tab_d.items()}

            optimizer.zero_grad()

            if model_type == 'tabular_only':
                preds, _, _ = model(tab_d)
            elif model_type == 'image_only':
                preds, _, _ = model(images)
            else:
                tau_dict = get_tau_dict(cfg, epoch)
                preds, _, _ = model(images, tab_d, tau_dict)

            loss, _ = regression_loss(preds, targets, TARGET_COLS, delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg['training']['grad_clip'])
            optimizer.step()
            train_losses.append(loss.item())

        # ── validate ─────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, tab_d, targets in val_loader:
                images  = images.to(device)
                targets = targets.to(device)
                tab_d   = {k: v.to(device) for k, v in tab_d.items()}

                if model_type == 'tabular_only':
                    preds, _, _ = model(tab_d)
                elif model_type == 'image_only':
                    preds, _, _ = model(images)
                else:
                    tau_dict = get_tau_dict(cfg, epoch)
                    preds, _, _ = model(images, tab_d, tau_dict)

                loss, _ = regression_loss(preds, targets, TARGET_COLS, delta)
                val_losses.append(loss.item())

        mean_val = np.mean(val_losses)
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  "
                  f"Train={np.mean(train_losses):.4f}  Val={mean_val:.4f}")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return _evaluate_neural_model(model, model_type, val_loader, device, cfg)


def _evaluate_neural_model(model, model_type, val_loader, device, cfg):
    """Evaluate any neural model (baseline or main) and return metrics dict."""
    model.eval()
    all_preds_orig   = defaultdict(list)
    all_targets_orig = defaultdict(list)

    with torch.no_grad():
        for images, tab_d, targets in val_loader:
            images  = images.to(device)
            targets = targets.to(device)
            tab_d   = {k: v.to(device) for k, v in tab_d.items()}

            if model_type == 'tabular_only':
                preds, _, _ = model(tab_d)
            elif model_type == 'image_only':
                preds, _, _ = model(images)
            else:
                tau_dict = get_tau_dict(cfg, 0)
                preds, _, _ = model(images, tab_d, tau_dict)

            orig_preds = _preds_to_orig_space(preds)
            for i, col in enumerate(TARGET_COLS):
                t_orig = torch.expm1(targets[:, i].clamp(min=0))
                all_targets_orig[col].append(t_orig.cpu())
                all_preds_orig[col].append(orig_preds[col].cpu())

    metrics = {}
    for col in TARGET_COLS:
        y_true = torch.cat(all_targets_orig[col]).numpy()
        y_pred = torch.cat(all_preds_orig[col]).numpy()
        metrics[col] = _compute_regression_metrics(y_true, y_pred)

    preds_tensor = {k: torch.cat(v) for k, v in all_preds_orig.items()}
    metrics['CSR'] = calculate_csr_metrics(preds_tensor)
    return metrics


# ─── Individual Baseline Runners ─────────────────────────────────────────────

def train_and_evaluate_b2(train_loader, val_loader, cfg, device, epochs=None):
    """B2: Tabular-Only Neural baseline."""
    print("\n" + "=" * 60)
    print("B2: Neural Tabular Only")
    print("=" * 60)
    model = TabularOnlyBaseline(cfg).to(device)
    m = _train_neural_baseline(
        model, 'tabular_only', train_loader, val_loader, cfg, device, epochs)
    _print_metrics("B2: Neural Tabular Only", m)
    return m


def train_and_evaluate_b3(train_loader, val_loader, cfg, device, epochs=None):
    """B3: Image-Only Neural baseline."""
    print("\n" + "=" * 60)
    print("B3: Neural Image Only")
    print("=" * 60)
    model = ImageOnlyBaseline(cfg).to(device)
    m = _train_neural_baseline(
        model, 'image_only', train_loader, val_loader, cfg, device, epochs)
    _print_metrics("B3: Neural Image Only", m)
    return m


def train_and_evaluate_b4(train_loader, val_loader, cfg, device, epochs=None):
    """B4: Full Neural (No Conservation, No LTN)."""
    print("\n" + "=" * 60)
    print("B4: Full Neural (No Conservation, No LTN)")
    print("=" * 60)
    model = FullNeuralNoConservation(cfg).to(device)
    m = _train_neural_baseline(
        model, 'full', train_loader, val_loader, cfg, device, epochs)
    _print_metrics("B4: Full Neural (No Conservation)", m)
    return m


def train_and_evaluate_b5(train_loader, val_loader, cfg, device, epochs=None):
    """B5: Full Neural (With Conservation, No LTN)."""
    print("\n" + "=" * 60)
    print("B5: Full Neural (With Conservation, No LTN)")
    print("=" * 60)
    model = FullNeuralWithConservation(cfg).to(device)
    m = _train_neural_baseline(
        model, 'full', train_loader, val_loader, cfg, device, epochs)
    _print_metrics("B5: Full Neural (Conservation Only)", m)
    return m


def evaluate_main_ltn_model(checkpoint_path, val_loader, cfg, device):
    """
    Load the trained main BiomassLTN model from checkpoint and evaluate it
    on the same val set used for baselines.
    """
    from model import BiomassLTNModel
    print("\n" + "=" * 60)
    print("Main: BiomassLTN (Full Neuro-Symbolic)")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print("  WARNING: checkpoint not found – skipping main model evaluation")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BiomassLTNModel(cfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    print("  Loaded checkpoint from", checkpoint_path)

    m = _evaluate_neural_model(model, 'full', val_loader, device, cfg)
    _print_metrics("Main: BiomassLTN", m)
    return m


# ─── Comparative Reporting ───────────────────────────────────────────────────

def build_comparison_table(all_results, output_dir=None):
    """
    Build a DataFrame comparing all models, optionally save CSV + text report.
    """
    rows = []
    for model_name, metrics in all_results.items():
        if metrics is None:
            continue
        for col in TARGET_COLS:
            if col in metrics:
                for mname, mval in metrics[col].items():
                    rows.append({'Model': model_name, 'Target': col,
                                 'Metric': mname, 'Value': mval})
        if 'CSR' in metrics:
            for cname, cval in metrics['CSR'].items():
                rows.append({'Model': model_name, 'Target': 'CSR',
                             'Metric': cname, 'Value': cval})

    df = pd.DataFrame(rows)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'comparative_analysis.csv'),
                  index=False)
        _save_comparative_summary(all_results, output_dir)

    return df


def _save_comparative_summary(all_results, output_dir):
    """Save a human-readable comparative summary text file."""
    path = os.path.join(output_dir, 'comparative_analysis.txt')
    model_names = [n for n, m in all_results.items() if m is not None]

    with open(path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("COMPARATIVE ANALYSIS - Baseline vs Neuro-Symbolic (LTN)\n")
        f.write("=" * 90 + "\n\n")

        for col in TARGET_COLS:
            f.write(f"--- {col} ---\n")
            f.write(f"{'Model':<50} {'RMSE':>8} {'MAE':>8} "
                    f"{'R2':>8} {'MAPE':>8}\n")
            f.write("-" * 90 + "\n")
            for name in model_names:
                m = all_results[name].get(col)
                if m:
                    f.write(f"{name:<50} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
                            f"{m['R2']:>8.4f} {m['MAPE']:>8.2f}\n")
            f.write("\n")

        # CSR table
        f.write("--- Constraint Satisfaction Rates ---\n")
        csr_keys = None
        for name in model_names:
            csr = all_results[name].get('CSR')
            if csr and csr_keys is None:
                csr_keys = list(csr.keys())
                hdr = f"{'Model':<50}"
                for k in csr_keys:
                    hdr += f" {k[:20]:>20}"
                f.write(hdr + "\n")
                f.write("-" * (50 + 21 * len(csr_keys)) + "\n")
            if csr:
                row = f"{name:<50}"
                for k in csr_keys:
                    row += f" {csr.get(k, float('nan')):>20.4f}"
                f.write(row + "\n")
        f.write("\n")

    print(f"\n  Comparative report saved -> {path}")


# ─── Orchestrator ────────────────────────────────────────────────────────────

def run_all_baselines(cfg, device, train_loader, val_loader,
                      train_df, val_df, output_dir,
                      baseline_epochs=None):
    """
    Train every baseline, evaluate the main LTN model, and produce
    a full comparative analysis.

    Returns (all_results, comparison_df).
    """
    metrics_dir = os.path.join(output_dir, 'metrics')
    ckpt_path   = os.path.join(output_dir, 'checkpoints', 'best_model.pt')

    all_results = {}

    # B1
    all_results['B1: XGBoost (Tabular + Rules)'] = \
        train_and_evaluate_xgboost(train_df, val_df, cfg)

    # B2
    all_results['B2: Neural Tabular Only'] = \
        train_and_evaluate_b2(train_loader, val_loader,
                              cfg, device, baseline_epochs)

    # B3
    all_results['B3: Neural Image Only'] = \
        train_and_evaluate_b3(train_loader, val_loader,
                              cfg, device, baseline_epochs)

    # B4
    all_results['B4: Full Neural (No Constraints)'] = \
        train_and_evaluate_b4(train_loader, val_loader,
                              cfg, device, baseline_epochs)

    # B5
    all_results['B5: Full Neural (Conservation Only)'] = \
        train_and_evaluate_b5(train_loader, val_loader,
                              cfg, device, baseline_epochs)

    # Main LTN model from checkpoint
    main_m = evaluate_main_ltn_model(ckpt_path, val_loader, cfg, device)
    if main_m is not None:
        all_results['Main: BiomassLTN (Full Neuro-Symbolic)'] = main_m

    # Build reports
    comparison_df = build_comparison_table(all_results, metrics_dir)

    print("\n" + "=" * 60)
    print("  COMPARATIVE ANALYSIS COMPLETE")
    print("=" * 60)

    return all_results, comparison_df


# ─── Standalone entry point ──────────────────────────────────────────────────

def main():
    """Run the full baseline comparative study from the command line."""
    import random

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = cfg['data'].get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(base_dir, 'outputs')

    csv_path = os.path.join(base_dir, cfg['data']['train_csv'])
    img_root = os.path.join(base_dir, cfg['data']['img_root'])

    df_wide, _ = load_and_preprocess(csv_path, img_root, cfg)
    train_df, val_df = split_data(df_wide, cfg)
    train_df, val_df, _ = fit_scaler(train_df, val_df, outputs_dir)

    train_loader, val_loader = build_dataloaders(train_df, val_df, cfg, epoch=0)

    baseline_epochs = cfg['training'].get('baseline_epochs',
                                          cfg['training']['epochs'])

    run_all_baselines(cfg, device, train_loader, val_loader,
                      train_df, val_df, outputs_dir, baseline_epochs)


if __name__ == '__main__':
    main()
