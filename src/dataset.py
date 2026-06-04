"""
BiomassDataset - Data loading, preprocessing, and collation for LTN biomass prediction.

Handles:
  - Long-to-wide CSV pivot (1785 rows -> 357 images x 5 targets)
  - log1p target transformation
  - Height log-transform + StandardScaler on numerics
  - Species/State/Month/Season label encoding
  - Structural zero flags for LTN predicates
  - Train/val split with joint Species+State stratification
  - Image augmentation (train) and normalization (val)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ─── Constants ───────────────────────────────────────────────────────────────

TARGET_COLS = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
PRIMARY_TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
DERIVED_TARGETS = ['GDM_g', 'Dry_Total_g']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Species with exactly zero clover (100% zero rate in dataset)
CLOVER_ZERO_SPECIES = {'Fescue', 'Lucerne', 'Phalaris', 'Mixed'}

# Species that are pure-clover (Dead=0, Green=0, 100%)
PURE_CLOVER_SPECIES = {'SubcloverLosa', 'SubcloverDalkeith'}

# State with structural zero dead biomass
DEAD_ZERO_STATE = {'WA'}

# Season mapping (Southern Hemisphere)
SEASON_MAP = {
    12: 'Summer', 1: 'Summer', 2: 'Summer',
    3:  'Autumn',  4: 'Autumn', 5: 'Autumn',
    6:  'Winter',  7: 'Winter', 8: 'Winter',
    9:  'Spring', 10: 'Spring', 11: 'Spring',
}


# ─── Label Encoders ──────────────────────────────────────────────────────────

class LabelEncoder:
    """Simple label encoder that maps string categories to integer indices."""

    def __init__(self):
        self.classes_ = []
        self._mapping = {}

    def fit(self, values):
        self.classes_ = sorted(set(values))
        self._mapping = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, values):
        return [self._mapping.get(v, 0) for v in values]

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    @property
    def n_classes(self):
        return len(self.classes_)


# ─── Preprocessing ───────────────────────────────────────────────────────────

def load_and_preprocess(csv_path, img_root, cfg):
    """
    Load train.csv (long format), pivot to wide, apply transforms.

    Returns:
        df_wide: DataFrame with columns for each target + metadata + structural flags
        label_encoders: dict of fitted LabelEncoders for categoricals
        scaler: fitted StandardScaler for numerical features
    """
    df = pd.read_csv(csv_path)

    # ── Pivot long -> wide ────────────────────────────────────────────────
    df_wide = df.pivot_table(
        index=['image_path', 'Sampling_Date', 'State', 'Species',
               'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        values='target'
    ).reset_index()

    # Flatten column names if MultiIndex
    df_wide.columns.name = None

    # ── Temporal features ────────────────────────────────────────────────
    df_wide['date'] = pd.to_datetime(df_wide['Sampling_Date'])
    df_wide['month'] = df_wide['date'].dt.month
    df_wide['season'] = df_wide['month'].map(SEASON_MAP)

    # ── Resolve image paths ──────────────────────────────────────────────
    df_wide['image_path_full'] = df_wide['image_path'].apply(
        lambda p: os.path.join(img_root, p)
    )

    # ── Height log-transform (before scaling) ────────────────────────────
    if cfg['data'].get('log_height', True):
        df_wide['Height_Ave_cm'] = np.log1p(df_wide['Height_Ave_cm'])

    # ── Target log-transform ─────────────────────────────────────────────
    if cfg['data'].get('use_log_targets', True):
        for col in TARGET_COLS:
            df_wide[col] = np.log1p(df_wide[col])

    # ── Structural zero flags ────────────────────────────────────────────
    # Check for species containing clover-zero keywords
    clover_zero_species_cfg = set(cfg['data'].get('clover_zero_species', CLOVER_ZERO_SPECIES))
    pure_clover_species_cfg = set(cfg['data'].get('pure_clover_species', PURE_CLOVER_SPECIES))
    dead_zero_state_cfg = set(cfg['data'].get('dead_zero_state', DEAD_ZERO_STATE))

    # Use substring matching for species that may appear as part of compound names
    def _is_clover_zero(species):
        """Check if species is in clover-zero category (handles compound names)."""
        for cz in clover_zero_species_cfg:
            if species == cz or species.startswith(cz + '_'):
                return True
        return False

    def _is_pure_clover(species):
        """Check if species is in pure-clover category."""
        return species in pure_clover_species_cfg

    df_wide['is_clover_zero'] = df_wide['Species'].apply(_is_clover_zero)
    df_wide['is_pure_clover'] = df_wide['Species'].apply(_is_pure_clover)
    df_wide['is_dead_zero_state'] = df_wide['State'].isin(dead_zero_state_cfg)

    # ── Label encoding ───────────────────────────────────────────────────
    label_encoders = {}
    for col_name in ['Species', 'State', 'season']:
        le = LabelEncoder()
        df_wide[f'{col_name}_idx'] = le.fit_transform(df_wide[col_name].values)
        label_encoders[col_name] = le

    # Month is already integer (1–12), encode to 0-indexed
    df_wide['month_idx'] = df_wide['month'] - 1

    return df_wide, label_encoders


def split_data(df_wide, cfg):
    """
    Stratified train/val split on joint Species+State key.

    Returns:
        train_df, val_df
    """
    df_wide['strat_key'] = df_wide['Species'] + '__' + df_wide['State']

    # Handle rare combinations: merge groups with < 2 samples into 'rare'
    counts = df_wide['strat_key'].value_counts()
    rare_keys = counts[counts < 2].index
    strat_col = df_wide['strat_key'].copy()
    strat_col[strat_col.isin(rare_keys)] = '__rare__'

    train_df, val_df = train_test_split(
        df_wide,
        test_size=cfg['data'].get('val_split', 0.20),
        stratify=strat_col,
        random_state=cfg['data'].get('random_seed', 42),
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def fit_scaler(train_df, val_df, output_dir):
    """
    Fit StandardScaler on training set numerics, apply to both splits.

    Returns:
        train_df, val_df (modified in-place), scaler
    """
    num_features = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
    scaler = StandardScaler()
    train_df[num_features] = scaler.fit_transform(train_df[num_features])
    val_df[num_features] = scaler.transform(val_df[num_features])

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    return train_df, val_df, scaler


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_train_transforms():
    return T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=15),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms():
    return T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─── Dataset ─────────────────────────────────────────────────────────────────

class BiomassDataset(Dataset):
    """
    PyTorch Dataset for biomass prediction.

    Each sample returns:
        image:  [3, 224, 224] tensor
        tab:    dict of tabular features (categorical indices + numerical + flags)
        targets: [5] tensor of target values (in log1p space if configured)
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Image ────────────────────────────────────────────────────────
        img_path = row['image_path_full']
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # ── Tabular features ─────────────────────────────────────────────
        tab = {
            'species': torch.tensor(row['Species_idx'], dtype=torch.long),
            'state':   torch.tensor(row['State_idx'],   dtype=torch.long),
            'month':   torch.tensor(row['month_idx'],   dtype=torch.long),
            'season':  torch.tensor(row['season_idx'],  dtype=torch.long),
            'numerical': torch.tensor(
                [row['Pre_GSHH_NDVI'], row['Height_Ave_cm']],
                dtype=torch.float32
            ),
            # Structural flags for LTN predicates
            'is_clover_zero':  torch.tensor(row['is_clover_zero'],    dtype=torch.bool),
            'is_pure_clover':  torch.tensor(row['is_pure_clover'],    dtype=torch.bool),
            'is_wa':           torch.tensor(row['is_dead_zero_state'], dtype=torch.bool),
            'is_winter':       torch.tensor(row['season'] == 'Winter', dtype=torch.bool),
            'is_summer':       torch.tensor(row['season'] == 'Summer', dtype=torch.bool),
        }

        # ── Targets ──────────────────────────────────────────────────────
        targets = torch.tensor(
            [row[col] for col in TARGET_COLS],
            dtype=torch.float32
        )

        return image, tab, targets


def biomass_collate_fn(batch):
    """
    Custom collate that stacks images/targets and batches tabular dicts.
    """
    images, tabs, targets = zip(*batch)

    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)

    # Stack tabular dict
    tab_batch = {}
    for key in tabs[0].keys():
        tab_batch[key] = torch.stack([t[key] for t in tabs], dim=0)

    return images, tab_batch, targets


# ─── Curriculum Learning ─────────────────────────────────────────────────────

def get_curriculum_weights(df, epoch, warmup_epochs=30):
    """
    Stage 1 (epoch < warmup): Downweight high-biomass samples.
    Stage 2 (epoch >= warmup): Full uniform sampling.

    Returns:
        weights: [N] tensor for WeightedRandomSampler, or None for uniform.
    """
    if epoch >= warmup_epochs:
        return None

    # In log-space: Q75 of Dry_Total_g (already log-transformed)
    q75 = df['Dry_Total_g'].quantile(0.75)
    progress = epoch / warmup_epochs
    weights = torch.ones(len(df))
    high_biomass = (df['Dry_Total_g'] > q75).values
    # High-biomass samples start at weight=0.1, increase to 1.0 by warmup
    weights[high_biomass] = 0.1 + 0.9 * progress
    return weights


def build_dataloaders(train_df, val_df, cfg, epoch=0):
    """
    Build train and val DataLoaders with optional curriculum weighting.
    """
    batch_size = cfg['training'].get('batch_size', 16)
    num_workers = cfg['training'].get('num_workers', 0)

    train_ds = BiomassDataset(train_df, transform=get_train_transforms())
    val_ds   = BiomassDataset(val_df,   transform=get_val_transforms())

    # Curriculum weighting
    sampler = None
    shuffle = True
    if cfg['training'].get('curriculum', {}).get('enabled', True):
        warmup = cfg['training']['curriculum'].get('warmup_epochs', 30)
        weights = get_curriculum_weights(train_df, epoch, warmup)
        if weights is not None:
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=biomass_collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=biomass_collate_fn,
        pin_memory=False,
    )

    return train_loader, val_loader
