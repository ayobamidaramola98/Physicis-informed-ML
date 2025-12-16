#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sfe.py

Comprehensive physics-informed + ML pipeline for predicting stacking fault
energy (SFE) in Cr–Fe–Mn–Ni–Co FCC alloys.

Key improvements over previous version:
 - Adds base_id per original composition
 - Dirichlet augmentation preserves base_id
 - Uses GroupKFold for CV on augmented data to prevent data leakage
 - Group-wise holdout split and conformal calibration

Author: Ayobami Daramola
Date: 2025-12-06
"""
from __future__ import annotations
import os
import sys
import math
import json
import time
import random
import warnings
import shutil
import pickle
import glob
import argparse
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Defensive imports & packages
# -----------------------------
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import (
        train_test_split,
        KFold,
        GroupKFold,
        cross_val_predict,
        cross_val_score,
        GridSearchCV
    )
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge as SkRidge
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
    from sklearn.inspection import permutation_importance, PartialDependenceDisplay
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.ensemble import StackingRegressor
    from sklearn.base import clone
    import xgboost as xgb
    import lightgbm as lgb
    import optuna
    import joblib
    import importlib
    import sklearn
    import scipy
    from scipy.cluster import hierarchy
    from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception as e:
    print("Critical import error. Ensure required packages are installed (numpy,pandas,scikit-learn,matplotlib,seaborn,xgboost,lightgbm,optuna).")
    print("Error:", repr(e))
    raise

# Optional explainability / deep learning
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Dataset = None
    DataLoader = None

# Optional packages (CALPHAD / DScribe SOAP)
try:
    import pycalphad as pc
    PCALPHAD_AVAILABLE = True
except Exception:
    PCALPHAD_AVAILABLE = False

try:
    from dscribe.descriptors import SOAP
    DSCRIBE_AVAILABLE = True
except Exception:
    DSCRIBE_AVAILABLE = False

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 36,

    "axes.labelsize": 36,
    "axes.titlesize": 36,

    "xtick.labelsize": 36,
    "ytick.labelsize": 36,

    "legend.fontsize": 36,
    "figure.titlesize": 36,

    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
})
mpl.rcParams["figure.figsize"] = (10, 8)
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300


# -----------------------------
# Global settings
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

RESULTS_DIR = "results_sfe_full"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
LOG_PATH = os.path.join(RESULTS_DIR, "run_log.txt")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

sns.set(style="whitegrid", context="notebook", font_scale=1.05)
matplotlib.rcParams['figure.dpi'] = 150

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------
# Embedded dataset (user-provided)
# -----------------------------
RAW_DATA = [
    [0,90,0,0,10,19],
    [0,84,0,0,16,32],
    [0,82,0,0,18,27],
    [0,80,0,0,20,21],
    [0,78,0,0,22,16],
    [0,75,0,0,25,16],
    [0,85,0,0,15,12],
    [0,80,0,0,20,18],
    [0,75,0,0,25,23],
    [0,70,0,0,30,24],
    [0,60,0,0,40,34],
    [0,0,0,98,2,124],
    [0,0,0,97,3,110],
    [0,0,0,95,5,113],
    [0,0,0,80,20,86],
    [0,0,0,100,0,125],
    [0,50,0,50,0,90],
    [25,25,25,25,0,30],
    [25,25,25,25,0,27],
    [0,20,20,40,20,69],
    [14,20,20,26,20,35],
    [26,20,20,14,20,23],
    [23.5,23.5,23.5,23.5,6,28],
    [21.5,21.5,21.5,21.5,14,29],
    [20,20,15,25,20,38],
    [21.22,21.233,21.22,21.22,15,21],
    [19.99,20.001,19.99,19.99,19.99,26],
    [18.74,18.762,18.74,18.74,24.99,36],
    [20,21.233,21.22,21.22,15,21],
    [20,20,20,20,20,26.5],
    [33.33,0,33.34,33.33,0,24],
    [33,0,38,29,0,12],
    [21.1,26.3,0,26.3,26.3,55],
    [19.5,24.4,0,24.4,31.7,65],
    [18.3,22.7,0,29.5,29.5,68],
    [16.6,20.8,0,31.3,31.3,72],
    [18,27,0,28,27,26],
    [18,65,0,10,7,21],
    [17,65,0,12,6,18],
    [19,66,0,8,7,23],
    [18,64,0,7,6,19],
    [16,72,0,12,0,24],
    [18,70,0,12,0,30],
    [20,68,0,12,0,13],
    [10,70,0,20,0,18],
    [0,85.2076,0,0,14.7924,25],
    [0,80.2607,0,0,19.7393,28],
    [0,75.3058,0,0,24.6942,27],
    [0,70.3427,0,0,29.6573,42],
    [18,68,0,14,0,32.2],
    [18,68,0,14,0,35],
    [18,68,0,14,0,35.2],
    [18,68,0,14,0,42.1],
    [18,68,0,14,0,34],
    [17,73,0,11,0,16],
    [17,67,0,16,0,28],
    [17,62,0,21,0,35],
    [14,75,0,11,0,16],
    [19,71,0,11,0,20],
    [24,66,0,11,0,25],
    [9,70,0,21,0,60],
    [14,65,0,21,0,40],
    [19,60,0,21,0,35],
    [23,55,0,21,0,45],
    [26,20,20,14,20,23],
    [15,68,0,0,17,25],
    [19.3,69.5,0,11.2,0,12.7],
    [17.3,71.7,0,11.0,0,17.7],
    [16.0,71.9,0,12.1,0,22.2],
    [13.0,72.8,0,14.2,0,33.1],
    [10.4,73.4,0,16.2,0,42.6],
    [19.2551,69.5063,0,11.2387,0,13.0],
    [17.2592,71.7291,0,11.0117,0,18.0],
    [16.0171,71.9295,0,12.0534,0,22.0],
    [12.9756,72.5882,0,14.4362,0,33.0],
    [10.4267,73.39,0,16.1833,0,43.0],
    [18.0,72.4,0,9.6,0,40.7],
    [18.0,67.9,0,14.1,0,40.9],
    [18.0,62.8,0,19.2,0,48.8],
    [18.0,68.0,0,14.0,0,32.3],
    [18.0,68.0,0,14.0,0,35.0],
    [18.0,68.0,0,14.0,0,35.2],
    [18.0,68.0,0,14.0,0,42.1],
    [18.0,68.0,0,14.0,0,34.0],
    [16.8828,72.5298,0,10.5874,0,16.0],
    [16.8395,67.3201,0,15.8404,0,28.0],
    [16.7965,62.137,0,21.0666,0,35.0],
    [14.0397,75.3949,0,10.5654,0,16.0],
    [18.7847,70.6132,0,10.6021,0,20.0],
    [23.5628,65.7981,0,10.6391,0,25.0],
    [9.28008,69.7691,0,20.9508,0,60.0],
    [13.9681,65.0089,0,21.023,0,40.0],
    [18.6885,60.2158,0,21.0957,0,35.0],
    [23.4417,55.3893,0,21.1689,0,45.0],
    [0,85.2076,0,0,14.7924,25.0],
    [0,80.2607,0,0,19.7393,28.0],
    [0,75.3058,0,0,24.6942,27.0],
    [0,70.3427,0,0,29.6573,42.0],
    [15,68,0,0,17,24],
    [18,56,0,16,10,44],
    [100,0,0,0,0,350],
    [0,100,0,0,0,1.0],
    [0,72,8,0,20,18.1],
    [0,76,5,0,19,12.9],
    [0,79,2,0,19,9.2],
    [0,80,0,0,20,7.0],
    [0,0,0,100,0,0.15],
    [0,0,91,9,0,5.92],
    [0,0,91,0,9,4.57],
    [0,9,91,0,0,28.33],
    [20,20,20,20,20,30],
    [20,20,20,20,20,26.5],
    [20,20,20,20,20,35],
    [0,25,25,25,25,59],
    [0,25,25,25,25,75],
    [25,0,25,25,25,32],
    [25,25,25,25,0,20],
    [25,25,25,25,0,27],
    [33.34,0,33.33,33.33,0,14],
    [33.34,0,33.33,33.33,0,18],
    [33.33,33.34,0,33.33,0,32],
    [33.33,33.34,0,33.33,0,36],
    [0,0,33.33,33.33,33.34,55],
    [0,33.33,0,33.34,33.33,83],
    [0,33.34,33.33,33.33,0,71],
    [0,50,0,50,0,100],
    [0,50,0,50,0,102],
    [0,0,50,50,0,50],
    [0,0,50,50,0,44],
    [20,72,0,8,0,14],
    [20,69,0,11,0,16.8],
    [20,69,0,11,0,14.0],
    [20,69,0,11,0,21.5],
    [20,69,0,11,0,16.9],
    [20,66,0,14,0,18.9],
    [20,66,0,14,0,26.5],
    [20,66,0,14,0,30.3],
    [20,66,0,14,0,37.3],
    [20,66,0,14,0,23.3],
    [20,66,0,14,0,25.8],
    [20,66,0,14,0,18.8],
    [20,66,0,14,0,32.8],
    [20,63,0,17,0,21.0],
    [20,60,0,20,0,25.0],
    [20,60,0,20,0,26.5],
    [20,60,0,20,0,26.9],
    [20,60,0,20,0,30.0],
    [20,60,0,20,0,23.8],
    [20,60,0,20,0,22.4],
    [20,60,0,20,0,19.3],
    [20,60,0,20,0,25.5],
    [20,50,0,30,0,29.4],
    [20,50,0,30,0,35.4],
    [20,50,0,30,0,39.0],
    [20,50,0,30,0,31.8],
    [20,50,0,30,0,30.9],
    [20,50,0,30,0,34.5],
    [20,50,0,30,0,27.3],
    [20,64,0,8,8,37.0],
    [20,61,0,11,8,28.0],
    [20,58,0,14,8,26.0],
    [20,58,0,14,8,26.1],
    [20,58,0,14,8,31.4],
    [20,58,0,14,8,20.8],
    [20,58,0,14,8,21.6],
    [20,58,0,14,8,26.9],
    [20,58,0,14,8,16.3],
    [20,55,0,17,8,26.5],
    [20,52,0,20,8,28.5],
    [20,52,0,20,8,34.8],
    [20,52,0,20,8,32.4],
    [20,52,0,20,8,37.2],
    [20,52,0,20,8,30.3],
    [20,52,0,20,8,32.7],
    [20,52,0,20,8,27.9],
    [20,49.5,0,22.5,8,33.0],
    [20,47,0,25,8,48.5],
    [20,42,0,30,8,54.0],
    [21.1,26.3,0,26.3,26.3,52.0],
    [21.1,26.3,0,26.3,26.3,42.0],
    [21.1,26.3,0,26.3,26.3,62.0],
    [19.5,24.4,0,24.4,31.7,65.0],
    [19.5,24.4,0,24.4,31.7,50.0],
    [19.5,24.4,0,24.4,31.7,70.0],
    [18.3,22.7,0,29.5,29.5,65.0],
    [18.3,22.7,0,29.5,29.5,60.0],
    [18.3,22.7,0,29.5,29.5,70.0],
    [16.6,20.8,0,31.3,31.3,70.0],
    [16.6,20.8,0,31.3,31.3,75.0],
    [16.6,20.8,0,31.3,31.3,65.0],
    [18,27.0,0,28.0,27.0,20.2],
    [18,27.0,0,28.0,27.0,26.8],
    [18,27.0,0,28.0,27.0,13.5],
    [15,35.0,0,35.0,15.0,9.2],
    [15,35.0,0,35.0,15.0,12.6],
    [15,35.0,0,35.0,15.0,5.8],
    [20.0,20.0,20.0,20.0,20.0,27.3],
    [20.0,20.0,20.0,20.0,20.0,25.3],
    [20.0,20.0,20.0,20.0,20.0,18.3],
    [20.0,20.0,20.0,20.0,20.0,19.6],
    [14.0,20.0,20.0,26.0,20.0,57.7],
    [14.0,20.0,20.0,23.0,20.0,19.7],
    [26.0,20.0,20.0,14.0,20.0,3.5],
    [21.5,21.5,21.5,14.0,21.5,7.7],
    [25.0,25.0,25.0,25.0,0.0,17.4],
    [25.0,25.0,25.0,25.0,0.0,34.3],
    [25.0,25.0,25.0,25.0,0.0,31.7],
    [0.0,0.0,0.0,100.0,0.0,121],
    [0.0,50.0,0.0,50.0,0.0,100],
    [33.33,33.34,0.0,33.33,0.0,60]
]
COLUMN_NAMES = ['Cr', 'Fe', 'Co', 'Ni', 'Mn', 'SFE_mJ_per_m2']

# -----------------------------
# Elemental properties (descriptors)
# -----------------------------
ELEMENTS = ['Cr', 'Fe', 'Co', 'Ni', 'Mn']
ELEMENT_PROPS = {
    'Cr': {'radius_pm': 128.0, 'en': 1.66, 'atomic_mass': 52.00, 'valence': 6},
    'Fe': {'radius_pm': 126.0, 'en': 1.83, 'atomic_mass': 55.85, 'valence': 8},
    'Co': {'radius_pm': 125.0, 'en': 1.88, 'atomic_mass': 58.93, 'valence': 9},
    'Ni': {'radius_pm': 124.0, 'en': 1.91, 'atomic_mass': 58.69, 'valence': 10},
    'Mn': {'radius_pm': 127.0, 'en': 1.55, 'atomic_mass': 54.94, 'valence': 7}
}

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str) -> None:
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_df(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(DATA_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def print_and_log(s: str) -> None:
    print(s)
    with open(LOG_PATH, "a") as f:
        f.write(s + "\n")

ensure_dir(RESULTS_DIR)
ensure_dir(FIG_DIR)
ensure_dir(MODEL_DIR)
ensure_dir(DATA_DIR)

# -----------------------------
# Data ingestion & normalization
# -----------------------------
def load_and_normalize_data() -> pd.DataFrame:
    df = pd.DataFrame(RAW_DATA, columns=COLUMN_NAMES)
    for c in COLUMN_NAMES:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    def norm_row(row):
        comps = np.array([row['Cr'], row['Fe'], row['Co'], row['Ni'], row['Mn']])
        s = comps.sum()
        if s <= 0:
            return pd.Series({'Cr':0,'Fe':0,'Co':0,'Ni':0,'Mn':0})
        if abs(s - 100.0) > 1e-6:
            comps = comps / s * 100.0
        return pd.Series({'Cr':comps[0],'Fe':comps[1],'Co':comps[2],'Ni':comps[3],'Mn':comps[4]})

    comps = df.apply(norm_row, axis=1)
    df[['Cr','Fe','Co','Ni','Mn']] = comps[['Cr','Fe','Co','Ni','Mn']]
    for el in ELEMENTS:
        df[f'{el}_frac'] = df[el] / 100.0

    save_df(df, "dataset_normalized.csv")
    return df

# -----------------------------
# Descriptor engineering
# -----------------------------
def compute_descriptors(df: pd.DataFrame, include_calphad: bool = False) -> pd.DataFrame:
    rows = []
    R_const = 8.31446261815324
    for idx, r in df.iterrows():
        fracs = np.array([r[f"{el}_frac"] for el in ELEMENTS], dtype=float)
        radii = np.array([ELEMENT_PROPS[el]['radius_pm'] for el in ELEMENTS])
        avg_radius = float(np.dot(fracs, radii))
        var_radius = float(np.dot(fracs, (radii - avg_radius)**2))
        delta_percent = float(np.sqrt(np.dot(fracs, ((radii - avg_radius)/avg_radius)**2)) * 100.0) if avg_radius>0 else 0.0
        ens = np.array([ELEMENT_PROPS[el]['en'] for el in ELEMENTS])
        avg_en = float(np.dot(fracs, ens))
        var_en = float(np.dot(fracs, (ens - avg_en)**2))
        vals = np.array([ELEMENT_PROPS[el]['valence'] for el in ELEMENTS])
        VEC = float(np.dot(fracs, vals))
        masses = np.array([ELEMENT_PROPS[el]['atomic_mass'] for el in ELEMENTS])
        avg_mass = float(np.dot(fracs, masses))

        smix = 0.0
        for p in fracs:
            if p > 0:
                smix -= p * np.log(p)
        S_mix = float(R_const * smix)

        Omega = 0.0
        for i in range(len(ELEMENTS)):
            for j in range(i+1, len(ELEMENTS)):
                xi = fracs[i]
                xj = fracs[j]
                omega_ij = abs(ens[i] - ens[j]) * (masses[i] + masses[j]) / 100.0
                Omega += 4.0 * xi * xj * omega_ij

        avg_pair_en_diff = float(np.mean([abs(ens[i] - ens[j]) for i in range(len(ELEMENTS)) for j in range(i+1, len(ELEMENTS))]))
        n_present = int((fracs > 0.01).sum())
        max_frac = float(fracs.max())
        min_nonzero = float(fracs[fracs > 0].min()) if fracs.sum() > 0 else 0.0

        dG_fcc_hcp = None
        if include_calphad and PCALPHAD_AVAILABLE:
            try:
                dG_fcc_hcp = None
            except Exception:
                dG_fcc_hcp = None

        row = {
            'Cr_frac': fracs[0],
            'Fe_frac': fracs[1],
            'Co_frac': fracs[2],
            'Ni_frac': fracs[3],
            'Mn_frac': fracs[4],
            'VEC': VEC,
            'avg_radius_pm': avg_radius,
            'var_radius_pm2': var_radius,
            'delta_size_percent': delta_percent,
            'avg_en': avg_en,
            'var_en': var_en,
            'avg_pair_en_diff': avg_pair_en_diff,
            'avg_mass': avg_mass,
            'S_mix_J_molK': S_mix,
            'mixing_enthalpy_proxy': Omega,
            'n_present': n_present,
            'max_frac': max_frac,
            'min_nonzero_frac': min_nonzero,
            'dG_fcc_hcp': dG_fcc_hcp,
            'SFE_mJ_per_m2': r['SFE_mJ_per_m2'],
            # base_id: unique ID per original composition (critical for grouped CV)
            'base_id': int(idx)
        }
        rows.append(row)

    df_desc = pd.DataFrame(rows)
    save_df(df_desc, "dataset_descriptors_full.csv")
    return df_desc

# -----------------------------
# Data augmentation (Dirichlet) with base_id preserved
# -----------------------------
def dirichlet_augment(df_desc: pd.DataFrame, n_per_point: int = 6, alpha: float = 100.0) -> pd.DataFrame:
    base_cols = ['Cr_frac', 'Fe_frac', 'Co_frac', 'Ni_frac', 'Mn_frac']
    augmented = []
    for _, r in df_desc.iterrows():
        base = np.array([r[c] for c in base_cols], dtype=float)
        base_id = int(r['base_id'])
        pseudo = base * alpha + 1e-8

        for _ in range(n_per_point):
            sample = np.random.dirichlet(pseudo)
            radii = np.array([ELEMENT_PROPS[el]['radius_pm'] for el in ELEMENTS])
            avg_radius = float(np.dot(sample, radii))
            var_radius = float(np.dot(sample, (radii - avg_radius)**2))
            delta_percent = float(np.sqrt(np.dot(sample, ((radii - avg_radius)/avg_radius)**2)) * 100.0) if avg_radius>0 else 0.0
            ens = np.array([ELEMENT_PROPS[el]['en'] for el in ELEMENTS])
            avg_en = float(np.dot(sample, ens))
            var_en = float(np.dot(sample, (ens - avg_en)**2))
            vals = np.array([ELEMENT_PROPS[el]['valence'] for el in ELEMENTS])
            VEC = float(np.dot(sample, vals))
            masses = np.array([ELEMENT_PROPS[el]['atomic_mass'] for el in ELEMENTS])
            avg_mass = float(np.dot(sample, masses))

            smix = 0.0
            R_const = 8.31446261815324
            for p in sample:
                if p > 0:
                    smix -= p * np.log(p)
            S_mix = float(R_const * smix)

            Omega = 0.0
            for i in range(len(ELEMENTS)):
                for j in range(i+1, len(ELEMENTS)):
                    xi = sample[i]
                    xj = sample[j]
                    omega_ij = abs(ens[i] - ens[j]) * (masses[i] + masses[j]) / 100.0
                    Omega += 4.0 * xi * xj * omega_ij

            avg_pair_en_diff = float(np.mean([abs(ens[i] - ens[j]) for i in range(len(ELEMENTS)) for j in range(i+1, len(ELEMENTS))]))
            n_present = int((sample > 0.01).sum())
            max_frac = float(sample.max())
            min_nonzero = float(sample[sample>0].min()) if sample.sum()>0 else 0.0

            row = {
                'Cr_frac': sample[0],
                'Fe_frac': sample[1],
                'Co_frac': sample[2],
                'Ni_frac': sample[3],
                'Mn_frac': sample[4],
                'VEC': VEC,
                'avg_radius_pm': avg_radius,
                'var_radius_pm2': var_radius,
                'delta_size_percent': delta_percent,
                'avg_en': avg_en,
                'var_en': var_en,
                'avg_pair_en_diff': avg_pair_en_diff,
                'avg_mass': avg_mass,
                'S_mix_J_molK': S_mix,
                'mixing_enthalpy_proxy': Omega,
                'n_present': n_present,
                'max_frac': max_frac,
                'min_nonzero_frac': min_nonzero,
                'dG_fcc_hcp': None,
                'SFE_mJ_per_m2': r['SFE_mJ_per_m2'],
                'base_id': base_id
            }
            augmented.append(row)

    if len(augmented) == 0:
        return df_desc.copy()

    df_aug = pd.DataFrame(augmented)
    df_combined = pd.concat([df_desc, df_aug], ignore_index=True).reset_index(drop=True)
    save_df(df_combined, "dataset_augmented_dirichlet.csv")
    return df_combined

# -----------------------------
# Metric helpers
# -----------------------------
def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    return {'RMSE': float(rmse), 'MAE': float(mae), 'R2': float(r2), 'MedAE': float(medae)}

# -----------------------------
# Embedding helper
# -----------------------------
def compute_embedding_2d(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = SEED) -> Tuple[np.ndarray, str]:
    try:
        from umap import UMAP as _UMAP
        reducer = _UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        emb = reducer.fit_transform(X)
        return emb, "umap-learn (from umap import UMAP)"
    except Exception:
        pass
    try:
        _umap_module = importlib.import_module('umap.umap_')
        _UMAP = getattr(_umap_module, 'UMAP', None)
        if _UMAP is not None:
            reducer = _UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
            emb = reducer.fit_transform(X)
            return emb, "umap-learn (umap.umap_.UMAP)"
    except Exception:
        pass
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=random_state, init='pca')
    emb = tsne.fit_transform(X)
    return emb, "sklearn TSNE (fallback)"

# -----------------------------
# Physics baseline + residual ML hybrid (with GroupKFold support)
# -----------------------------
def train_hybrid(df: pd.DataFrame,
                 feature_cols: List[str],
                 physics_cols: List[str],
                 xgb_params: Optional[dict] = None,
                 do_cv: bool = True,
                 groups: Optional[np.ndarray] = None) -> dict:
    X = df[feature_cols].values.astype(float)
    y = df['SFE_mJ_per_m2'].values.astype(float)
    X_phys = df[physics_cols].values.astype(float)

    scaler_phys = StandardScaler().fit(X_phys)
    Xp_scaled = scaler_phys.transform(X_phys)
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xp_scaled, y)
    phys_pred = ridge.predict(Xp_scaled)
    residuals = y - phys_pred

    if xgb_params is None:
        xgb_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 800,
            'random_state': SEED,
            'n_jobs': 4
        }
    xgb_resid = xgb.XGBRegressor(**xgb_params)
    xgb_resid.fit(X, residuals)

    resid_pred = xgb_resid.predict(X)
    hybrid_pred = phys_pred + resid_pred
    metrics_train = metrics_dict(y, hybrid_pred)

    cv_preds = None
    metrics_cv = None
    if do_cv:
        if groups is not None:
            cv = GroupKFold(n_splits=5)
            y_cv_pred = np.zeros_like(y)
            for train_idx, val_idx in cv.split(X, y, groups=groups):
                Xp_tr = Xp_scaled[train_idx]
                y_tr = y[train_idx]
                ridge_cv = Ridge(alpha=1.0).fit(Xp_tr, y_tr)
                phys_val = ridge_cv.predict(Xp_scaled[val_idx])

                resid_tr = y_tr - ridge_cv.predict(Xp_tr)
                xgb_cv = xgb.XGBRegressor(**xgb_params)
                xgb_cv.fit(X[train_idx], resid_tr)
                resid_val_pred = xgb_cv.predict(X[val_idx])

                y_cv_pred[val_idx] = phys_val + resid_val_pred
            cv_preds = y_cv_pred
            metrics_cv = metrics_dict(y, cv_preds)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            y_cv_pred = np.zeros_like(y)
            for train_idx, val_idx in cv.split(X):
                Xp_tr = Xp_scaled[train_idx]
                y_tr = y[train_idx]
                ridge_cv = Ridge(alpha=1.0).fit(Xp_tr, y_tr)
                phys_val = ridge_cv.predict(Xp_scaled[val_idx])

                resid_tr = y_tr - ridge_cv.predict(Xp_tr)
                xgb_cv = xgb.XGBRegressor(**xgb_params)
                xgb_cv.fit(X[train_idx], resid_tr)
                resid_val_pred = xgb_cv.predict(X[val_idx])

                y_cv_pred[val_idx] = phys_val + resid_val_pred
            cv_preds = y_cv_pred
            metrics_cv = metrics_dict(y, cv_preds)

    joblib.dump(ridge, os.path.join(MODEL_DIR, "ridge_physics.joblib"))
    joblib.dump(scaler_phys, os.path.join(MODEL_DIR, "scaler_phys.joblib"))
    try:
        xgb_resid.save_model(os.path.join(MODEL_DIR, "xgb_residual_model.json"))
    except Exception:
        with open(os.path.join(MODEL_DIR, "xgb_residual_model.pkl"), "wb") as f:
            pickle.dump(xgb_resid, f)

    return {
        'ridge': ridge,
        'scaler_phys': scaler_phys,
        'xgb_resid': xgb_resid,
        'phys_pred_train': phys_pred,
        'resid_pred_train': resid_pred,
        'hybrid_pred_train': hybrid_pred,
        'metrics_train': metrics_train,
        'cv_preds': cv_preds,
        'metrics_cv': metrics_cv,
        'feature_cols': feature_cols,
        'physics_cols': physics_cols
    }

# -----------------------------
# Baselines & ensembling (with GroupKFold support)
# -----------------------------
def train_baselines(X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> dict:
    results = {}
    if groups is not None:
        cv = GroupKFold(n_splits=5)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf_scores = cross_val_score(
        rf, X, y, cv=cv, scoring='r2', n_jobs=1,
        groups=groups if groups is not None else None
    )
    rf.fit(X, y)
    results['RF'] = {'model': rf, 'cv_r2': float(rf_scores.mean()), 'cv_r2_all': rf_scores}
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_baseline.joblib"))

    # XGBoost
    try:
        xgbm = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500, random_state=SEED, n_jobs=4)
        xgb_scores = cross_val_score(
            xgbm, X, y, cv=cv, scoring='r2', n_jobs=1,
            groups=groups if groups is not None else None
        )
        xgbm.fit(X, y)
        results['XGB'] = {'model': xgbm, 'cv_r2': float(xgb_scores.mean()), 'cv_r2_all': xgb_scores}
        try:
            xgbm.save_model(os.path.join(MODEL_DIR, "xgb_baseline.json"))
        except Exception:
            joblib.dump(xgbm, os.path.join(MODEL_DIR, "xgb_baseline.joblib"))
    except Exception as e:
        print("XGBoost baseline failed:", e)

    # LightGBM
    lgbm = lgb.LGBMRegressor(n_estimators=500, random_state=SEED, n_jobs=4)
    lgbm_scores = cross_val_score(
        lgbm, X, y, cv=cv, scoring='r2', n_jobs=1,
        groups=groups if groups is not None else None
    )
    lgbm.fit(X, y)
    results['LGBM'] = {'model': lgbm, 'cv_r2': float(lgbm_scores.mean()), 'cv_r2_all': lgbm_scores}
    joblib.dump(lgbm, os.path.join(MODEL_DIR, "lgbm_baseline.joblib"))

    return results

# -----------------------------
# Optional PyTorch MLP + MC Dropout
# -----------------------------
if TORCH_AVAILABLE:
    class SfeDatasetTorch(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    class MLP_MC(nn.Module):
        def __init__(self, in_dim: int, hidden: Tuple[int,int] = (128,128), dropout: float = 0.2):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden[0])
            self.do1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden[0], hidden[1])
            self.do2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(hidden[1], 1)
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
            nn.init.xavier_uniform_(self.fc3.weight)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.do1(x)
            x = F.relu(self.fc2(x))
            x = self.do2(x)
            return self.fc3(x)

    def train_mlp_torch(X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, params: Optional[dict] = None):
        params = params or {}
        n_features = X_train.shape[1]
        hidden = params.get('hidden', (128,128))
        dropout = params.get('dropout', 0.2)
        lr = params.get('lr', 1e-3)
        wd = params.get('weight_decay', 1e-5)
        batch_size = params.get('batch_size', 32)
        n_epochs = params.get('n_epochs', 400)
        patience = params.get('patience', 40)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MLP_MC(n_features, hidden=hidden, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()
        train_ds = SfeDatasetTorch(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = SfeDatasetTorch(X_val, y_val)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        best_state = None
        best_val = float('inf')
        no_improve = 0
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*xb.size(0)
            avg_train = total_loss/len(train_ds)
            if val_loader:
                model.eval()
                total_val = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device); yb = yb.to(device)
                        preds = model(xb)
                        total_val += criterion(preds, yb).item()*xb.size(0)
                avg_val = total_val/len(val_ds)
                if avg_val < best_val - 1e-6:
                    best_val = avg_val
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > patience:
                    break
            else:
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if epoch % 50 == 0 or epoch == n_epochs-1:
                print(f"MLP epoch {epoch}/{n_epochs}, train_loss={avg_train:.6f}, val_loss={locals().get('avg_val', None)}")
        if best_state:
            model.load_state_dict(best_state)
        return model

    def mc_dropout_predict_torch(model: nn.Module, X: np.ndarray, n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        model.train()
        device = next(model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = model(X_tensor).cpu().numpy().reshape(-1)
                preds.append(out)
        preds = np.stack(preds, axis=0)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    def conformal_prediction_residuals(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        resid = np.abs(y_true - y_pred)
        q = np.quantile(resid, 1 - alpha)
        lower = y_pred - q
        upper = y_pred + q
        return lower, upper
else:
    def train_mlp_torch(*args, **kwargs):
        raise RuntimeError("PyTorch not available.")
    def mc_dropout_predict_torch(*args, **kwargs):
        raise RuntimeError("PyTorch not available.")
    def conformal_prediction_residuals(*args, **kwargs):
        raise RuntimeError("PyTorch not available.")

# -----------------------------
# Explainability helpers
# -----------------------------
def compute_shap_tree_and_save(model, X_sample: np.ndarray, feature_names: List[str], out_png: str) -> Optional[Any]:
    if not SHAP_AVAILABLE:
        print("SHAP not available; skipping tree SHAP summary.")
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, features=X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved SHAP summary: {out_png}")
        return shap_values
    except Exception as e:
        print("SHAP TreeExplainer failed:", e)
        return None

def permutation_importance_and_save(model, X: np.ndarray, y: np.ndarray, feature_names: List[str], out_png: str) -> None:
    try:
        res = permutation_importance(model, X, y, n_repeats=16, random_state=SEED, n_jobs=1)
        imp = pd.Series(res.importances_mean, index=feature_names).sort_values(ascending=True)
        plt.figure(figsize=(7,6))
        imp.tail(20).plot(kind='barh')
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        imp.to_csv(out_png.replace('.png', '.csv'))
        print(f"Saved permutation importance and csv: {out_png}")
    except Exception as e:
        print("Permutation importance failed:", e)

# -----------------------------
# Plotting helpers (improved feature correlation)
# -----------------------------
def plot_feature_corr_improved(df: pd.DataFrame, cols: List[str], out_png_base: str) -> None:
    """
    Improved feature correlation visualization:
      - hierarchical reorder heatmap
      - mask upper triangle to reduce clutter
      - compact clustermap with dendrograms
    """
    data = df[cols].copy()
    corr = data.corr()

    # Hierarchical clustering to reorder features (by absolute correlation)
    try:
        dist = 1.0 - np.abs(corr.values)
        from scipy.spatial.distance import squareform
        linkage = hierarchy.linkage(squareform(dist), method='average')
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        order = dendro['leaves']
        ordered_cols = [cols[i] for i in order]
    except Exception:
        ordered_cols = cols

    # Mask upper triangle so annotations don't overlap
    ordered_corr = corr.loc[ordered_cols, ordered_cols]
    mask = np.triu(np.ones_like(ordered_corr, dtype=bool), k=1)

    # ------------------ Heatmap ---------------------
    plt.figure(figsize=(24, 20))
    ax = sns.heatmap(
        ordered_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={'size': 26},
        cmap='vlag',
        center=0,
        square=True,
        linewidths=1.2,
        cbar_kws={'label': 'Pearson r'}
    )

    # tick label font
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=28)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=28)

    # colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label("Pearson r", fontsize=28)

    plt.tight_layout()
    out_reorder = out_png_base.replace('.png', '_reordered.png')
    plt.savefig(out_reorder, dpi=300)
    plt.close()
    print(f"Saved reordered feature correlation: {out_reorder}")

    # ------------------ Clustermap ------------------
    try:
        cg = sns.clustermap(
            corr,
            method='average',
            cmap='vlag',
            center=0,
            figsize=(12, 9),
            annot=False,
            cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}
        )
        # increase font of ticklabels if you want (optional):
        for lbl in cg.ax_heatmap.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_fontsize(26)
        for lbl in cg.ax_heatmap.get_yticklabels():
            lbl.set_fontsize(26)

        out_clustermap = out_png_base.replace('.png', '_clustermap.png')
        cg.savefig(out_clustermap, dpi=300)
        plt.close()
        print(f"Saved clustermap: {out_clustermap}")
    except Exception as e:
        print("Clustermap generation failed:", e)


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, out_png: str, title: str = "Parity plot") -> None:
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=y_true, y=y_pred, s=50, edgecolor='k', linewidth=0.3)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
    plt.xlabel("Measured SFE (mJ/m²)")
    plt.ylabel("Predicted SFE (mJ/m²)")
    plt.xlim(mn - 5, mx + 5)
    plt.ylim(mn - 5, mx + 5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved parity: {out_png}")

def residuals_plot(y_true: np.ndarray, y_pred: np.ndarray, out_png: str) -> None:
    resid = y_true - y_pred
    plt.figure(figsize=(7,6))
    sns.scatterplot(x=y_pred, y=resid, s=50, edgecolor='k', linewidth=0.3)
    plt.axhline(0, linestyle='--', color='k')
    plt.xlabel("Predicted SFE (mJ/m²)")
    plt.ylabel("Residual (Measured - Predicted)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved residuals: {out_png}")

def kde_compare(y_true: np.ndarray, y_pred: np.ndarray, out_png: str) -> None:
    plt.figure(figsize=(7,6))
    sns.kdeplot(y_true, label='Measured', fill=False)
    sns.kdeplot(y_pred, label='Predicted', fill=False)
    plt.xlabel("SFE (mJ/m²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved KDE compare: {out_png}")

# -----------------------------
# Advanced models: GPR, KRR, Stacking (kept as in original)
# -----------------------------
def composition_rbf_kernel(X, Y=None, length_scale=1.0):
    if Y is None:
        Y = X
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    XX = np.sum(X*X, axis=1)[:, None]
    YY = np.sum(Y*Y, axis=1)[None, :]
    D2 = XX + YY - 2.0 * np.dot(X, Y.T)
    K = np.exp(-0.5 * D2 / (length_scale**2))
    return K

def train_gpr_model(X, y, n_restarts_optimizer=5, normalize_y=True):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5) \
             + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e2))
    try:
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y)
        gpr.fit(X, y)
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_preds = cross_val_predict(gpr, X, y, cv=cv, n_jobs=1)
        metrics = metrics_dict(y, cv_preds)
        return gpr, metrics, cv_preds
    except Exception as e:
        print("GPR training failed:", e)
        return None, None, None

def train_gpr_with_composition_kernel(X, y, comp_indices: List[int], length_scale=0.5, n_restarts_optimizer=3):
    try:
        X_comp = X[:, comp_indices]
        scaler_comp = StandardScaler().fit(X_comp)
        Xc = scaler_comp.transform(X_comp)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, normalize_y=True)
        gpr.fit(Xc, y)
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_preds = np.zeros_like(y)
        for train_idx, val_idx in cv.split(Xc):
            gpr_cv = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            gpr_cv.fit(Xc[train_idx], y[train_idx])
            cv_preds[val_idx] = gpr_cv.predict(Xc[val_idx])
        metrics = metrics_dict(y, cv_preds)
        return {'gpr': gpr, 'scaler_comp': scaler_comp, 'metrics': metrics, 'cv_preds': cv_preds}
    except Exception as e:
        print('Composition GPR failed:', e)
        return None

def train_krr_grid(X, y, alphas=(1e-3, 1e-2, 1e-1, 1, 10), gammas=(1e-3,1e-2,1e-1,1), cv=4):
    param_grid = {'alpha': list(alphas), 'gamma': list(gammas)}
    kr = KernelRidge(kernel='rbf')
    gs = GridSearchCV(kr, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=4, verbose=0)
    try:
        gs.fit(X, y)
        best = gs.best_estimator_
        cv_eval = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_preds = cross_val_predict(best, X, y, cv=cv_eval, n_jobs=1)
        metrics = metrics_dict(y, cv_preds)
        return best, metrics, cv_preds, gs.best_params_
    except Exception as e:
        print("KRR GridSearch failed:", e)
        return None, None, None, None

def build_and_train_stacking(X, y, base_models: dict, meta_estimator=None, cv=5):
    if meta_estimator is None:
        meta_estimator = SkRidge(alpha=1.0)
    estimators = [(name, clone(m)) for name,m in base_models.items()]
    stack = StackingRegressor(estimators=estimators, final_estimator=meta_estimator, cv=cv, n_jobs=4, passthrough=False)
    try:
        stack.fit(X, y)
        cv_preds = cross_val_predict(stack, X, y, cv=cv, n_jobs=1)
        metrics = metrics_dict(y, cv_preds)
        return stack, metrics, cv_preds
    except Exception as e:
        print("Stacking training failed:", e)
        return None, None, None

def select_best_model_by_rmse(candidate_results: dict):
    best_name = None
    best_rmse = float('inf')
    best_r2 = -float('inf')
    for name, info in candidate_results.items():
        m = info.get('metrics')
        if m is None:
            continue
        rmse = m.get('RMSE', float('inf'))
        r2 = m.get('R2', -float('inf'))
        if rmse < best_rmse - 1e-9 or (abs(rmse - best_rmse) < 1e-9 and r2 > best_r2):
            best_rmse = rmse
            best_r2 = r2
            best_name = name
    return best_name, candidate_results.get(best_name)

def save_model_pickle(model, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model to {path}")
    except Exception as e:
        print("Failed to save model:", e)

def run_advanced_models(df, FEATURE_COLS, PHYSICS_COLS, hybrid_results, baselines):
    X = df[FEATURE_COLS].values.astype(float)
    y = df['SFE_mJ_per_m2'].values.astype(float)

    scaler_rob = RobustScaler().fit(X)
    X_scaled = scaler_rob.transform(X)

    candidates = {}

    print("Training GPR (full features, Matern kernel)...")
    gpr, gpr_metrics, gpr_cv_preds = train_gpr_model(X_scaled, y, n_restarts_optimizer=5)
    if gpr is not None:
        candidates['GPR_full'] = {'model': gpr, 'metrics': gpr_metrics, 'cv_preds': gpr_cv_preds}
        save_model_pickle(gpr, os.path.join(MODEL_DIR, "gpr_model_full.pkl"))

    try:
        comp_idx = [FEATURE_COLS.index(c) for c in ['Cr_frac','Fe_frac','Co_frac','Ni_frac','Mn_frac']]
        print("Training composition-only GPR (RBF kernel on composition fractions)...")
        gpr_comp_res = train_gpr_with_composition_kernel(X, y, comp_idx, length_scale=0.5, n_restarts_optimizer=3)
        if gpr_comp_res is not None:
            candidates['GPR_comp'] = {'model': gpr_comp_res['gpr'], 'metrics': gpr_comp_res['metrics'], 'cv_preds': gpr_comp_res['cv_preds'], 'scaler_comp': gpr_comp_res['scaler_comp']}
            save_model_pickle(gpr_comp_res['gpr'], os.path.join(MODEL_DIR, "gpr_model_comp.pkl"))
    except Exception as e:
        print("Composition GPR step failed:", e)

    print("Training Kernel Ridge Regression (KRR) with RBF kernel grid search...")
    krr, krr_metrics, krr_cv_preds, krr_best = train_krr_grid(X_scaled, y, alphas=(1e-3,1e-2,1e-1,1,10), gammas=(1e-3,1e-2,1e-1,1))
    if krr is not None:
        candidates['KRR'] = {'model': krr, 'metrics': krr_metrics, 'cv_preds': krr_cv_preds, 'best_params': krr_best}
        save_model_pickle(krr, os.path.join(MODEL_DIR, "krr_model.pkl"))

    for name, d in baselines.items():
        try:
            model = d['model']
            cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
            preds_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            metrics = metrics_dict(y, preds_cv)
            candidates[name] = {'model': model, 'metrics': metrics, 'cv_preds': preds_cv}
            print(f"Baseline {name} CV metrics: {metrics}")
        except Exception as e:
            print(f"Baseline {name} eval failed: {e}")

    try:
        xgb_model = hybrid_results.get('xgb_resid', None)
        if xgb_model is not None:
            preds_cv = cross_val_predict(xgb_model, X, y - hybrid_results['phys_pred_train'], cv=KFold(n_splits=5,shuffle=True,random_state=SEED), n_jobs=1)
            phys = hybrid_results['phys_pred_train']
            full_cv_preds = phys + preds_cv
            metrics = metrics_dict(y, full_cv_preds)
            candidates['XGB_fullhybrid'] = {'model': xgb_model, 'metrics': metrics, 'cv_preds': full_cv_preds}
            print("XGB (hybrid residual) CV metrics on full SFE:", metrics)
    except Exception as e:
        print("XGBoost evaluation in advanced pipeline failed:", e)

    stack_bases = {}
    for k in ['RF', 'XGB', 'LGBM', 'KRR', 'GPR_comp', 'GPR_full', 'XGB_fullhybrid']:
        info = candidates.get(k)
        if info is not None:
            stack_bases[k] = info['model']
    if len(stack_bases) >= 2:
        print("Training stacking ensemble from:", list(stack_bases.keys()))
        stack_model, stack_metrics, stack_cv_preds = build_and_train_stacking(X, y, stack_bases, meta_estimator=SkRidge(alpha=1.0), cv=5)
        if stack_model is not None:
            candidates['Stacked'] = {'model': stack_model, 'metrics': stack_metrics, 'cv_preds': stack_cv_preds}
            save_model_pickle(stack_model, os.path.join(MODEL_DIR, "stacked_model.pkl"))
            print("Stacking ensemble metrics:", stack_metrics)

    best_name, best_info = select_best_model_by_rmse(candidates)
    if best_name is None:
        print("No candidate model available to select as best.")
        return candidates
    print(f"Selected best model: {best_name} with metrics {best_info['metrics']}")
    save_model_pickle(best_info['model'], os.path.join(MODEL_DIR, f"best_model_{best_name}.pkl"))

    metrics_summary = {name: info.get('metrics') for name, info in candidates.items()}
    with open(os.path.join(RESULTS_DIR, "advanced_models_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    for name, info in candidates.items():
        preds = info.get('cv_preds')
        if preds is not None:
            dfp = pd.DataFrame({'measured': y, 'pred_cv': preds})
            dfp.to_csv(os.path.join(DATA_DIR, f"cv_preds_{name}.csv"), index=False)

    best_preds = best_info.get('cv_preds')
    parity_plot(y, best_preds, os.path.join(FIG_DIR, f"parity_best_{best_name}.png"), title=f"Best model ({best_name}) CV Parity")
    residuals_plot(y, best_preds, os.path.join(FIG_DIR, f"resid_best_{best_name}.png"))

    return {'candidates': candidates, 'best': (best_name, best_info)}

# -----------------------------
# Optuna routines (XGB and MLP)
# -----------------------------
def optuna_xgb_tune(X: np.ndarray, y: np.ndarray, n_trials: int = 24) -> dict:
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'random_state': SEED,
            'n_jobs': 4
        }
        cv = KFold(n_splits=4, shuffle=True, random_state=SEED)
        scores = []
        for train_idx, val_idx in cv.split(X):
            model = xgb.XGBRegressor(**param)
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], early_stopping_rounds=50, verbose=False)
            preds = model.predict(X[val_idx])
            scores.append(r2_score(y[val_idx], preds))
        return np.mean(scores)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def optuna_mlp_tune(X: np.ndarray, y: np.ndarray, n_trials: int = 12) -> dict:
    if not TORCH_AVAILABLE:
        return {}
    def objective(trial):
        n1 = trial.suggest_int('n_units1', 32, 512)
        n2 = trial.suggest_int('n_units2', 16, 512)
        dropout = trial.suggest_uniform('dropout', 0.05, 0.5)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        params = {'hidden': (n1, n2), 'dropout': dropout, 'lr': lr, 'batch_size': batch_size, 'n_epochs': 200, 'patience': 30}
        cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
        r2s = []
        for train_idx, val_idx in cv.split(X):
            model = train_mlp_torch(X[train_idx], y[train_idx], X_val=X[val_idx], y_val=y[val_idx], params=params)
            mean, _ = mc_dropout_predict_torch(model, X[val_idx], n_samples=80)
            r2s.append(r2_score(y[val_idx], mean))
        return float(np.mean(r2s))
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

# -----------------------------
# New helper: centralized predictor
# -----------------------------
def get_sfe_pred_from_models(frac: np.ndarray, FEATURE_COLS: List[str], PHYSICS_COLS: List[str],
                             ridge=None, scaler_phys=None, xgb_resid=None, best_model=None) -> float:
    """
    Return predicted SFE (float or np.nan) for a given fraction array [Cr,Fe,Co,Ni,Mn] (sums to 1).
    Attempts hybrid (ridge + xgb_resid) first, then best_model fallback.
    """
    feat = _compute_feature_row_from_frac(frac, FEATURE_COLS)
    X_row = np.array([feat[c] for c in FEATURE_COLS]).reshape(1, -1)

    # try to load provided models if None
    try:
        if ridge is None:
            ridge = joblib.load(os.path.join(MODEL_DIR, "ridge_physics.joblib"))
    except Exception:
        ridge = None
    try:
        if scaler_phys is None:
            scaler_phys = joblib.load(os.path.join(MODEL_DIR, "scaler_phys.joblib"))
    except Exception:
        scaler_phys = None
    try:
        if xgb_resid is None:
            pjson = os.path.join(MODEL_DIR, "xgb_residual_model.json")
            ppkl = os.path.join(MODEL_DIR, "xgb_residual_model.pkl")
            if os.path.exists(pjson):
                x = xgb.XGBRegressor(); x.load_model(pjson); xgb_resid = x
            elif os.path.exists(ppkl):
                with open(ppkl, "rb") as f:
                    xgb_resid = pickle.load(f)
    except Exception:
        xgb_resid = None

    pred = None
    if ridge is not None and scaler_phys is not None and xgb_resid is not None:
        try:
            phys_val = float(ridge.predict(scaler_phys.transform(np.array([feat[c] for c in PHYSICS_COLS]).reshape(1,-1)))[0])
            resid_val = float(xgb_resid.predict(X_row)[0])
            pred = phys_val + resid_val
            return float(pred)
        except Exception:
            pred = None

    if best_model is None:
        bfiles = glob.glob(os.path.join(MODEL_DIR, "best_model_*.pkl")) + glob.glob(os.path.join(MODEL_DIR, "best_model_*.joblib")) + glob.glob(os.path.join(MODEL_DIR, "stacked_model.pkl"))
        if bfiles:
            try:
                best_model = joblib.load(bfiles[0])
            except Exception:
                best_model = None

    if best_model is not None:
        try:
            return float(best_model.predict(X_row)[0])
        except Exception:
            return float('nan')

    return float('nan')

# -----------------------------
# New: Tetrahedral 3D scatter for Fe–Ni–Cr–Mn (Co=0)
# -----------------------------
def generate_tetrahedral_FeNiCrMn_plot(FEATURE_COLS: List[str], PHYSICS_COLS: List[str],
                                      ridge=None, scaler_phys=None, xgb_resid=None, best_model=None,
                                      step: float = 0.08, save_prefix: str = "FeNiCrMn_tetra"):
    """
    Create a tetrahedral 3D scatter for compositions (Cr, Fe, Ni, Mn) with Co=0.
    Map barycentric coords to tetrahedron vertices and color by predicted SFE.
    """
    vCr = np.array([1.0, 1.0, 1.0])
    vFe = np.array([-1.0, -1.0, 1.0])
    vNi = np.array([-1.0, 1.0, -1.0])
    vMn = np.array([1.0, -1.0, -1.0])
    verts = np.vstack([vCr, vFe, vNi, vMn])

    comps, preds = [], []
    vals = np.arange(0.0, 1.0 + 1e-12, step)
    for cr in vals:
        for fe in vals:
            for ni in vals:
                for mn in vals:
                    s = cr + fe + ni + mn
                    if s <= 1.0 + 1e-12 and s > 0:
                        w = np.array([cr, fe, ni, mn]) / s
                        frac = np.array([w[0], w[1], 0.0, w[2], w[3]])
                        pred = get_sfe_pred_from_models(frac, FEATURE_COLS, PHYSICS_COLS, ridge=ridge, scaler_phys=scaler_phys, xgb_resid=xgb_resid, best_model=best_model)
                        comps.append(w)
                        preds.append(pred)

    if len(comps) == 0:
        print("No points generated for tetrahedron (step too large?)")
        return {'png': None, 'csv': None}

    comps = np.array(comps)
    preds = np.array(preds)
    coords = comps @ verts

    levels = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 350.0]
    cmap_colors = ["#08306B", "#2B8CBE", "#7FCDBB", "#C7E9B4", "#FFFFB2", "#FED976", "#FD8D3C", "#FC4E2A"]
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=preds, cmap=cmap, norm=norm, s=10, depthshade=True)
    tetra = np.array([vCr, vFe, vNi, vMn, vCr])
    ax.plot(tetra[[0,1,2,3,0],0], tetra[[0,1,2,3,0],1], tetra[[0,1,2,3,0],2], 'k-', lw=1.0)
    cbar = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Predicted SFE (mJ/m$^2$)')
    for name, v in zip(['Cr','Fe','Ni','Mn'], verts):
        ax.text(v[0]*1.05, v[1]*1.05, v[2]*1.05, name, fontsize=26, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    out_png = os.path.join(FIG_DIR, f"{save_prefix}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved tetrahedral Fe-Ni-Cr-Mn plot: {out_png}")
    df_pts = pd.DataFrame({
        'Cr_frac': comps[:,0], 'Fe_frac': comps[:,1], 'Ni_frac': comps[:,2], 'Mn_frac': comps[:,3], 'SFE_pred': preds
    })
    csv_out = os.path.join(DATA_DIR, f"{save_prefix}_points.csv")
    df_pts.to_csv(csv_out, index=False)
    print(f"Saved tetrahedral CSV points: {csv_out}")
    return {'png': out_png, 'csv': csv_out}

# -----------------------------
# Advanced plotting & grid generation helpers (Fe-Ni-Cr)
# -----------------------------
def generate_fe_ni_cr_grid(FEATURE_COLS: List[str], PHYSICS_COLS: List[str],
                           ridge=None, scaler_phys=None, xgb_resid=None, best_model=None,
                           step_fine: float = 0.01, step_coarse: float = 0.05,
                           save_prefix: str = "fe_ni_cr"):
    x_vals = np.arange(0.0, 1.0 + 1e-12, step_fine)
    y_vals = np.arange(0.0, 1.0 + 1e-12, step_fine)
    fine_rows = []
    for x in x_vals:
        for y in y_vals:
            if x + y <= 1.0 + 1e-12:
                ni = x
                cr = y
                fe = 1.0 - ni - cr
                frac = np.array([cr, fe, 0.0, ni, 0.0])
                pred = get_sfe_pred_from_models(frac, FEATURE_COLS, PHYSICS_COLS, ridge=ridge, scaler_phys=scaler_phys, xgb_resid=xgb_resid, best_model=best_model)
                fine_rows.append({'Cr_frac': cr, 'Ni_frac': ni, 'Fe_frac': fe, 'SFE_pred': (None if math.isnan(pred) else float(pred))})
    df_fine = pd.DataFrame(fine_rows)
    fine_csv = os.path.join(DATA_DIR, f"{save_prefix}_fine_grid.csv")
    df_fine.to_csv(fine_csv, index=False)
    print(f"Saved fine grid CSV: {fine_csv}")

    x_vals_c = np.arange(0.0, 1.0 + 1e-12, step_coarse)
    y_vals_c = np.arange(0.0, 1.0 + 1e-12, step_coarse)
    coarse_rows = []
    for x in x_vals_c:
        for y in y_vals_c:
            if x + y <= 1.0 + 1e-12:
                ni = x; cr = y; fe = 1.0 - ni - cr
                frac = np.array([cr, fe, 0.0, ni, 0.0])
                pred = get_sfe_pred_from_models(frac, FEATURE_COLS, PHYSICS_COLS, ridge=ridge, scaler_phys=scaler_phys, xgb_resid=xgb_resid, best_model=best_model)
                coarse_rows.append({'Cr_frac': cr, 'Ni_frac': ni, 'Fe_frac': fe, 'SFE_pred': (None if math.isnan(pred) else float(pred))})
    df_coarse = pd.DataFrame(coarse_rows)
    coarse_csv = os.path.join(DATA_DIR, f"{save_prefix}_coarse_grid.csv")
    df_coarse.to_csv(coarse_csv, index=False)
    print(f"Saved coarse grid CSV: {coarse_csv}")

    df_plot = df_fine.dropna(subset=['SFE_pred']).reset_index(drop=True)
    pts_x = df_plot['Ni_frac'].values * 100.0
    pts_y = df_plot['Cr_frac'].values * 100.0
    vals = df_plot['SFE_pred'].values
    out_png = None
    try:
        levels = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 350.0]
        cmap_colors = [
            "#0033CC",
            "#66CCFF",
            "#00FFFF",
            "#CCFF99",
            "#00CC66",
            "#FFFF00",
            "#FF9900",
            "#FF0000"
        ]
        cmap = ListedColormap(cmap_colors)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        tri = Triangulation(pts_x, pts_y)
        plt.figure(figsize=(7,6))
        tcf = plt.tricontourf(tri, vals, levels=levels, cmap=cmap, norm=norm, extend='both')
        plt.tricontour(tri, vals, levels=levels, colors='k', linewidths=0.6)
        xs = np.linspace(0,100,200)
        plt.plot(xs, 100.0 - xs, 'k-', lw=1.0)
        cbar = plt.colorbar(tcf, ticks=levels, spacing='proportional')
        cbar.set_label('Predicted SFE (mJ/m$^2$)')
        cbar.ax.set_yticklabels([str(int(l)) for l in levels])
        plt.xlabel("C_Ni (at.%)")
        plt.ylabel("C_Cr (at.%)")
        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        out_png = os.path.join(FIG_DIR, f"{save_prefix}_contour_pct.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved Fe-Ni-Cr contour plot (0-100 at.%): {out_png}")
    except Exception as e:
        print("Contour plotting failed:", e)

    return {'fine_csv': fine_csv, 'coarse_csv': coarse_csv, 'contour_png': out_png}

# -----------------------------
# New: Ternary-style plot for Cr-Mn-Ni (Fe = 1 - Cr - Mn - Ni)
# -----------------------------
def _ternary_to_cartesian(cr: np.ndarray, mn: np.ndarray, ni: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cr = np.asarray(cr)
    mn = np.asarray(mn)
    ni = np.asarray(ni)
    s = cr + mn + ni
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(s > 0, cr / s, 1.0)
        b = np.where(s > 0, mn / s, 0.0)
        c = np.where(s > 0, ni / s, 0.0)
    x = b + 0.5 * c
    y = (np.sqrt(3) / 2.0) * c
    return x, y

def generate_ternary_cr_mn_ni(FEATURE_COLS: List[str], PHYSICS_COLS: List[str],
                              ridge=None, scaler_phys=None, xgb_resid=None, best_model=None,
                              step: float = 0.02, save_prefix: str = "ternary_cr_mn_ni"):
    cr_list, mn_list, ni_list, fe_list, preds = [], [], [], [], []
    vals = np.arange(0.0, 1.0 + 1e-12, step)
    for cr in vals:
        for mn in vals:
            for ni in vals:
                if cr + mn + ni <= 1.0 + 1e-12:
                    fe = 1.0 - (cr + mn + ni)
                    frac = np.array([cr, fe, 0.0, ni, mn])
                    pred = get_sfe_pred_from_models(frac, FEATURE_COLS, PHYSICS_COLS, ridge=ridge, scaler_phys=scaler_phys, xgb_resid=xgb_resid, best_model=best_model)
                    cr_list.append(cr); mn_list.append(mn); ni_list.append(ni); fe_list.append(fe); preds.append(pred)
    df_tern = pd.DataFrame({'Cr_frac': cr_list, 'Mn_frac': mn_list, 'Ni_frac': ni_list, 'Fe_frac': fe_list, 'SFE_pred': preds})
    out_csv = os.path.join(DATA_DIR, f"{save_prefix}_points.csv")
    df_tern.to_csv(out_csv, index=False)
    print(f"Saved ternary candidate CSV: {out_csv}")

    df_plot = df_tern.dropna(subset=['SFE_pred']).reset_index(drop=True)
    x, y = _ternary_to_cartesian(df_plot['Cr_frac'].values, df_plot['Mn_frac'].values, df_plot['Ni_frac'].values)
    plt.figure(figsize=(7,6))
    sc = plt.scatter(x, y, c=df_plot['SFE_pred'].values, cmap='viridis', s=20, edgecolor='none')
    cbar = plt.colorbar(sc); cbar.set_label('Predicted SFE (mJ/m²)')
    triangle = np.array([[0.0,0.0], [1.0,0.0], [0.5, np.sqrt(3)/2.0], [0.0,0.0]])
    plt.plot(triangle[:,0], triangle[:,1], 'k-', lw=0.8)
    plt.axis('equal')
    plt.axis('off')
    out_png = os.path.join(FIG_DIR, f"{save_prefix}_ternary.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved ternary plot: {out_png}")

    return {'ternary_csv': out_csv, 'ternary_png': out_png}

# -----------------------------
# Prediction helpers
# -----------------------------
def _make_frac_vector_from_input(comp_in: dict) -> np.ndarray:
    comps = {el: 0.0 for el in ELEMENTS}
    for k,v in comp_in.items():
        if k not in comps:
            continue
        val = float(v)
        if val <= 1.0:
            comps[k] = val * 100.0
        else:
            comps[k] = val
    total = sum(comps.values())
    if total <= 0:
        raise ValueError("Composition sums to zero.")
    vec_pct = np.array([comps[el] for el in ELEMENTS], dtype=float) / total * 100.0
    return vec_pct / 100.0

def _compute_feature_row_from_frac(frac: np.ndarray, FEATURE_COLS: Optional[List[str]] = None) -> dict:
    radii = np.array([ELEMENT_PROPS[e]['radius_pm'] for e in ELEMENTS])
    ens = np.array([ELEMENT_PROPS[e]['en'] for e in ELEMENTS])
    vals = np.array([ELEMENT_PROPS[e]['valence'] for e in ELEMENTS])
    masses = np.array([ELEMENT_PROPS[e]['atomic_mass'] for e in ELEMENTS])
    avg_radius = float(np.dot(frac, radii))
    var_radius = float(np.dot(frac, (radii - avg_radius)**2))
    delta_percent = float(np.sqrt(np.dot(frac, ((radii - avg_radius)/avg_radius)**2))*100.0) if avg_radius>0 else 0.0
    avg_en = float(np.dot(frac, ens))
    var_en = float(np.dot(frac, (ens - avg_en)**2))
    VEC = float(np.dot(frac, vals))
    avg_mass = float(np.dot(frac, masses))
    smix = 0.0
    for p in frac:
        if p > 0:
            smix -= p * np.log(p)
    S_mix = float(8.31446261815324 * smix)
    Omega = 0.0
    for i in range(len(frac)):
        for j in range(i+1, len(frac)):
            xi = frac[i]; xj = frac[j]
            omega_ij = abs(ens[i] - ens[j]) * (masses[i] + masses[j]) / 100.0
            Omega += 4.0 * xi * xj * omega_ij
    feat = {
        'Cr_frac': frac[0], 'Fe_frac': frac[1], 'Co_frac': frac[2], 'Ni_frac': frac[3], 'Mn_frac': frac[4],
        'VEC': VEC, 'avg_radius_pm': avg_radius, 'var_radius_pm2': var_radius, 'delta_size_percent': delta_percent,
        'avg_en': avg_en, 'var_en': var_en, 'avg_pair_en_diff': float(np.mean([abs(ens[a]-ens[b]) for a in range(len(ens)) for b in range(a+1, len(ens))])),
        'avg_mass': avg_mass, 'S_mix_J_molK': S_mix, 'mixing_enthalpy_proxy': Omega, 'n_present': int((frac>0.01).sum()),
        'max_frac': float(frac.max()), 'min_nonzero_frac': float((frac[frac>0].min() if frac.sum()>0 else 0.0))
    }
    if FEATURE_COLS is None:
        return feat
    return {k: feat[k] for k in FEATURE_COLS}

def predict_composition(comp_in: dict, FEATURE_COLS: List[str], PHYSICS_COLS: List[str], model_preference: str = 'hybrid') -> dict:
    frac = _make_frac_vector_from_input(comp_in)
    feat = _compute_feature_row_from_frac(frac, FEATURE_COLS)
    X_feat_ordered = np.array([feat[c] for c in FEATURE_COLS]).reshape(1, -1)

    ridge = None; scaler_phys = None; xgb_resid = None
    try:
        ridge = joblib.load(os.path.join(MODEL_DIR, "ridge_physics.joblib"))
    except Exception:
        ridge = None
    try:
        scaler_phys = joblib.load(os.path.join(MODEL_DIR, "scaler_phys.joblib"))
    except Exception:
        scaler_phys = None

    xgb_path_json = os.path.join(MODEL_DIR, "xgb_residual_model.json")
    xgb_path_pkl = os.path.join(MODEL_DIR, "xgb_residual_model.pkl")
    try:
        if os.path.exists(xgb_path_json):
            xgb_resid = xgb.XGBRegressor()
            xgb_resid.load_model(xgb_path_json)
        elif os.path.exists(xgb_path_pkl):
            with open(xgb_path_pkl, "rb") as f:
                xgb_resid = pickle.load(f)
    except Exception:
        xgb_resid = None

    phys_val = None
    if ridge is not None and scaler_phys is not None:
        phys_input = np.array([feat[c] for c in PHYSICS_COLS]).reshape(1, -1)
        try:
            phys_val = float(ridge.predict(scaler_phys.transform(phys_input))[0])
        except Exception:
            phys_val = None

    resid_val = None
    if xgb_resid is not None:
        try:
            resid_val = float(xgb_resid.predict(X_feat_ordered)[0])
        except Exception:
            resid_val = None

    if phys_val is not None and resid_val is not None:
        pred = phys_val + resid_val
    elif resid_val is not None:
        pred = resid_val
    elif phys_val is not None:
        pred = phys_val
    else:
        pred = None

    if (model_preference == 'best' or pred is None):
        bf = glob.glob(os.path.join(MODEL_DIR, "best_model_*.pkl")) + glob.glob(os.path.join(MODEL_DIR, "best_model_*.joblib")) + glob.glob(os.path.join(MODEL_DIR, "stacked_model.pkl"))
        if bf:
            try:
                best_model = joblib.load(bf[0])
                pred_best = float(best_model.predict(X_feat_ordered)[0])
                if model_preference == 'best' or pred is None:
                    pred = pred_best
                    phys_val = None; resid_val = None
            except Exception:
                pass

    ensemble_std = None
    preds_ensemble = []
    for mname in ["rf_baseline.joblib", "xgb_baseline.joblib", "lgbm_baseline.joblib"]:
        fp = os.path.join(MODEL_DIR, mname)
        if os.path.exists(fp):
            try:
                mod = joblib.load(fp)
                preds_ensemble.append(float(mod.predict(X_feat_ordered)[0]))
            except Exception:
                pass
    if len(preds_ensemble) > 0:
        ensemble_std = float(np.std(preds_ensemble))
    q95 = None
    q95_file = os.path.join(MODEL_DIR, "q95_conformal.npy")
    if os.path.exists(q95_file):
        try:
            q95 = float(np.load(q95_file))
        except Exception:
            q95 = None
    else:
        hf = os.path.join(DATA_DIR, "holdout_with_intervals.csv")
        if os.path.exists(hf):
            try:
                hdf = pd.read_csv(hf)
                if 'measured' in hdf.columns and 'predicted' in hdf.columns:
                    nonconf = np.abs(hdf['measured'].values - hdf['predicted'].values)
                    q95 = float(np.quantile(nonconf, 0.95))
            except Exception:
                q95 = None

    interval = (None, None)
    if q95 is not None and pred is not None:
        interval = (float(pred - q95), float(pred + q95))

    return {
        'input_pct': {el: float(frac[i]*100.0) for i,el in enumerate(ELEMENTS)},
        'predicted_SFE_mJ_per_m2': (None if pred is None else float(pred)),
        'physics_contribution_mJ_per_m2': phys_val,
        'residual_contribution_mJ_per_m2': resid_val,
        'ensemble_std_mJ_per_m2': ensemble_std,
        'conformal_q95_mJ_per_m2': q95,
        'conformal_interval': interval,
        'used_model_preference': model_preference
    }

# -----------------------------
# Bayesian optimization wrapper (simple)
# -----------------------------
def bayesian_optimize_composition(FEATURE_COLS: List[str], PHYSICS_COLS: List[str], n_trials: int = 200, direction: str = 'minimize', lock_nonzero: Optional[dict] = None, objective_model: str = 'hybrid'):
    def predict_from_frac_array(frac_arr):
        feat = _compute_feature_row_from_frac(frac_arr, FEATURE_COLS)
        X_row = np.array([feat[c] for c in FEATURE_COLS]).reshape(1, -1)
        if objective_model == 'hybrid':
            try:
                ridge = joblib.load(os.path.join(MODEL_DIR, "ridge_physics.joblib"))
                scaler_phys = joblib.load(os.path.join(MODEL_DIR, "scaler_phys.joblib"))
                if os.path.exists(os.path.join(MODEL_DIR, "xgb_residual_model.json")):
                    xgb_resid = xgb.XGBRegressor(); xgb_resid.load_model(os.path.join(MODEL_DIR, "xgb_residual_model.json"))
                elif os.path.exists(os.path.join(MODEL_DIR, "xgb_residual_model.pkl")):
                    with open(os.path.join(MODEL_DIR, "xgb_residual_model.pkl"), "rb") as f:
                        xgb_resid = pickle.load(f)
                phys_val = float(ridge.predict(scaler_phys.transform(np.array([feat[c] for c in PHYSICS_COLS]).reshape(1,-1)))[0])
                resid_val = float(xgb_resid.predict(X_row)[0])
                return phys_val + resid_val
            except Exception:
                pass
        bfiles = glob.glob(os.path.join(MODEL_DIR, "best_model_*.pkl")) + glob.glob(os.path.join(MODEL_DIR, "best_model_*.joblib"))
        if bfiles:
            try:
                best_model = joblib.load(bfiles[0])
                return float(best_model.predict(X_row)[0])
            except Exception:
                pass
        return float('nan')

    def objective(trial):
        raw = np.array([trial.suggest_uniform(f"r{i}", 0.0, 1.0) for i in range(len(ELEMENTS))])
        if lock_nonzero:
            for i, el in enumerate(ELEMENTS):
                if el in lock_nonzero:
                    mn, mx = lock_nonzero[el]
                    raw[i] = trial.suggest_uniform(f"fixed_{el}", mn, mx)
        s = raw.sum()
        if s <= 0:
            frac = np.array([1.0/len(ELEMENTS)]*len(ELEMENTS))
        else:
            frac = raw / s
        val = predict_from_frac_array(frac)
        if math.isnan(val):
            return 1e6
        return float(val) if direction == 'minimize' else float(-val)

    study = optuna.create_study(direction='minimize' if direction == 'minimize' else 'maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial
    comp = {}
    for i, el in enumerate(ELEMENTS):
        k = f"r{i}"
        if k in best.params:
            comp[el] = best.params[k]
        elif f"fixed_{el}" in best.params:
            comp[el] = best.params[f"fixed_{el}"]
        else:
            comp[el] = 0.0
    s = sum([comp[el] for el in ELEMENTS])
    if s > 0:
        for el in ELEMENTS:
            comp[el] = comp[el] / s
    return {'best': comp, 'value': best.value, 'trial': best.number}

# -----------------------------
# Command-line interface (enhanced)
# -----------------------------
def _parse_comp_string(comp_str: str) -> dict:
    try:
        if comp_str.strip().startswith("{"):
            d = json.loads(comp_str)
            return {k.strip(): float(v) for k,v in d.items()}
        parts = comp_str.split()
        d = {}
        for p in parts:
            if '=' in p:
                k,v = p.split('=',1)
                d[k.strip()] = float(v)
        return d
    except Exception:
        raise ValueError("Failed to parse composition string. Use format: Fe=80 Mn=20 or JSON.")

def cli():
    parser = argparse.ArgumentParser(description="SFE pipeline CLI: predict, run, BO, generate grids/ternary, plot features.")
    sub = parser.add_subparsers(dest='cmd')

    p_pred = sub.add_parser('predict', help='Predict SFE for a given composition')
    p_pred.add_argument('--comp', type=str, required=True, help="Composition string, e.g. 'Fe=80 Mn=20' or JSON")
    p_pred.add_argument('--model', type=str, default='hybrid', choices=['hybrid','best'], help="Model preference")

    p_bo = sub.add_parser('bo', help='Run Bayesian optimization to find compositions minimizing/maximizing SFE')
    p_bo.add_argument('--trials', type=int, default=200, help='Number of Optuna trials')
    p_bo.add_argument('--direction', type=str, default='minimize', choices=['minimize','maximize'], help='Minimize or maximize SFE')
    p_bo.add_argument('--lock', type=str, default=None, help="Optional locks like 'Ni:0.0-0.4;Mn:0.0-0.3' (fractions)")

    p_run = sub.add_parser('run', help='Run the full training pipeline (train models and save artifacts)')

    p_fe = sub.add_parser('gen-fe-ni-cr', help='Generate Fe-Ni-Cr fine/coarse grids and contour plot')
    p_fe.add_argument('--fine_step', type=float, default=0.01, help='Fine grid fraction step (default 0.01)')
    p_fe.add_argument('--coarse_step', type=float, default=0.05, help='Coarse grid step (default 0.05)')

    p_tern = sub.add_parser('gen-ternary', help='Generate ternary Cr-Mn-Ni predictions and plot')
    p_tern.add_argument('--step', type=float, default=0.02, help='Ternary grid step (default 0.02)')

    p_feat = sub.add_parser('plot-feat-corr', help='Generate improved feature correlation figures')

    args = parser.parse_args()
    if args.cmd == 'predict':
        comp = _parse_comp_string(args.comp)
        FEATURE_COLS = ['Cr_frac','Fe_frac','Co_frac','Ni_frac','Mn_frac',
                        'VEC','avg_radius_pm','var_radius_pm2','delta_size_percent',
                        'avg_en','var_en','avg_pair_en_diff','avg_mass','S_mix_J_molK',
                        'mixing_enthalpy_proxy','n_present','max_frac','min_nonzero_frac']
        PHYSICS_COLS = ['VEC','delta_size_percent','S_mix_J_molK','mixing_enthalpy_proxy','avg_en','var_en']
        res = predict_composition(comp, FEATURE_COLS, PHYSICS_COLS, model_preference=args.model)
        print(json.dumps(res, indent=2))
    elif args.cmd == 'bo':
        FEATURE_COLS = ['Cr_frac','Fe_frac','Co_frac','Ni_frac','Mn_frac',
                        'VEC','avg_radius_pm','var_radius_pm2','delta_size_percent',
                        'avg_en','var_en','avg_pair_en_diff','avg_mass','S_mix_J_molK',
                        'mixing_enthalpy_proxy','n_present','max_frac','min_nonzero_frac']
        PHYSICS_COLS = ['VEC','delta_size_percent','S_mix_J_molK','mixing_enthalpy_proxy','avg_en','var_en']
        lock_dict = None
        if args.lock:
            lock_dict = {}
            items = args.lock.split(';')
            for it in items:
                if ':' in it:
                    el, rng = it.split(':',1)
                    mn,mx = rng.split('-',1)
                    lock_dict[el.strip()] = (float(mn), float(mx))
        print_and_log(f"Starting BO ({args.direction}) with {args.trials} trials, locks: {lock_dict}")
        out = bayesian_optimize_composition(FEATURE_COLS, PHYSICS_COLS, n_trials=args.trials, direction=args.direction, lock_nonzero=lock_dict, objective_model='hybrid')
        print("Best composition (fraction):")
        print(json.dumps(out, indent=2))
    elif args.cmd == 'run':
        main_pipeline()
    elif args.cmd == 'gen-fe-ni-cr':
        FEATURE_COLS = ['Cr_frac','Fe_frac','Co_frac','Ni_frac','Mn_frac',
                        'VEC','avg_radius_pm','var_radius_pm2','delta_size_percent',
                        'avg_en','var_en','avg_pair_en_diff','avg_mass','S_mix_J_molK',
                        'mixing_enthalpy_proxy','n_present','max_frac','min_nonzero_frac']
        PHYSICS_COLS = ['VEC','delta_size_percent','S_mix_J_molK','mixing_enthalpy_proxy','avg_en','var_en']
        ridge = None; scaler_phys = None; xgb_resid = None; best_model = None
        try:
            ridge = joblib.load(os.path.join(MODEL_DIR, "ridge_physics.joblib"))
            scaler_phys = joblib.load(os.path.join(MODEL_DIR, "scaler_phys.joblib"))
            if os.path.exists(os.path.join(MODEL_DIR, "xgb_residual_model.json")):
                xgb_resid = xgb.XGBRegressor(); xgb_resid.load_model(os.path.join(MODEL_DIR, "xgb_residual_model.json"))
            elif os.path.exists(os.path.join(MODEL_DIR, "xgb_residual_model.pkl")):
                with open(os.path.join(MODEL_DIR, "xgb_residual_model.pkl"), "rb") as f:
                    xgb_resid = pickle.load(f)
        except Exception:
            pass
        try:
            bfiles = glob.glob(os.path.join(MODEL_DIR, "best_model_*.pkl")) + glob.glob(os.path.join(MODEL_DIR, "best_model_*.joblib"))
            if bfiles:
                best_model = joblib.load(bfiles[0])
        except Exception:
            best_model = None
        gen_res = generate_fe_ni_cr_grid(FEATURE_COLS, PHYSICS_COLS, ridge=ridge, scaler_phys=scaler_phys, xgb_resid=xgb_resid, best_model=best_model, step_fine=args.fine_step, step_coarse=args.coarse_step)
        print(json.dumps(gen_res, indent=2))
    elif args.cmd == 'gen-ternary':
        FEATURE_COLS = ['Cr_frac','Fe_frac','Co_frac','Ni_frac','Mn_frac',
                        'VEC','avg_radius_pm','var_radius_pm2','delta_size_percent',
                        'avg_en','var_en','avg_pair_en_diff','avg_mass','S_mix_J_molK',
                        'mixing_enthalpy_proxy','n_present','max_frac','min_nonzero_frac']
        PHYSICS_COLS = ['VEC','delta_size_percent','S_mix_J_molK','mixing_enthalpy_proxy','avg_en','var_en']
        ridge = None; scaler_phys = None; xgb_resid = None; best_model = None
        try:
            ridge = joblib.load(os.path.join(MODEL_DIR, "ridge_physics.joblib"))
            scaler_phys = joblib.load(os.path.join(MODEL_DIR, "scaler_phys.joblib"))
            if os.path.exists(os.path.join(MODEL_DIR, "xgb_residual_model.json")):
                xgb_resid = xgb.XGBRegressor(); xgb_resid.load_model(os.path.join(MODEL_DIR, "xgb_residual_model.json"))
            elif os.path.exists(os.path.join(MODEL_DIR, "xgb_residual_model.pkl")):
                with open(os.path.join(MODEL_DIR, "xgb_residual_model.pkl"), "rb") as f:
                    xgb_resid = pickle.load(f)
        except Exception:
            pass
        try:
            bfiles = glob.glob(os.path.join(MODEL_DIR, "best_model_*.pkl")) + glob.glob(os.path.join(MODEL_DIR, "best_model_*.joblib"))
            if bfiles:
                best_model = joblib.load(bfiles[0])
        except Exception:
            best_model = None
        res = generate_ternary_cr_mn_ni(FEATURE_COLS, PHYSICS_COLS, ridge=ridge, scaler_phys=scaler_phys, xgb_resid=xgb_resid, best_model=best_model, step=args.step)
        print(json.dumps(res, indent=2))
    elif args.cmd == 'plot-feat-corr':
        try:
            df_desc = pd.read_csv(os.path.join(DATA_DIR, "dataset_descriptors_full.csv"))
        except Exception:
            df = load_and_normalize_data()
            df_desc = compute_descriptors(df, include_calphad=False)
        FEATURE_COLS = ['Cr_frac','Fe_frac','Co_frac','Ni_frac','Mn_frac',
                        'VEC','avg_radius_pm','var_radius_pm2','delta_size_percent',
                        'avg_en','var_en','avg_pair_en_diff','avg_mass','S_mix_J_molK',
                        'mixing_enthalpy_proxy','n_present','max_frac','min_nonzero_frac']
        out_png_base = os.path.join(FIG_DIR, "fig01_feature_corr_improved.png")
        plot_feature_corr_improved(df_desc, FEATURE_COLS, out_png_base)
    else:
        parser.print_help()

# -----------------------------
# Main pipeline
# -----------------------------
def main_pipeline():
    t0 = time.time()
    print_and_log("Starting SFE hybrid pipeline...")

    df = load_and_normalize_data()
    df_desc = compute_descriptors(df, include_calphad=False)
    df_aug = dirichlet_augment(df_desc, n_per_point=6, alpha=100.0)

    FEATURE_COLS = ['Cr_frac','Fe_frac','Co_frac','Ni_frac','Mn_frac',
                    'VEC','avg_radius_pm','var_radius_pm2','delta_size_percent',
                    'avg_en','var_en','avg_pair_en_diff','avg_mass','S_mix_J_molK',
                    'mixing_enthalpy_proxy','n_present','max_frac','min_nonzero_frac']
    PHYSICS_COLS = ['VEC','delta_size_percent','S_mix_J_molK','mixing_enthalpy_proxy','avg_en','var_en']

    for c in FEATURE_COLS:
        if c not in df_aug.columns:
            raise KeyError(f"Feature {c} missing in data")

    groups_all = df_aug['base_id'].values.astype(int)

    print_and_log("Training physics baseline + residual XGBoost hybrid (GroupKFold CV)...")
    hybrid = train_hybrid(df_aug, FEATURE_COLS, PHYSICS_COLS, xgb_params=None, do_cv=True, groups=groups_all)
    print_and_log(f"Hybrid training metrics: {hybrid['metrics_train']}")
    if hybrid['metrics_cv'] is not None:
        print_and_log(f"Hybrid GroupKFold CV metrics: {hybrid['metrics_cv']}")

    X_mat = df_aug[FEATURE_COLS].values.astype(float)
    y_vec = df_aug['SFE_mJ_per_m2'].values.astype(float)
    baselines = train_baselines(X_mat, y_vec, groups=groups_all)

    try:
        print_and_log("Running Optuna tuning for residual XGBoost (small budget)...")
        tuned_params = optuna_xgb_tune(X_mat, y_vec - hybrid['phys_pred_train'], n_trials=24)
        print_and_log(f"Optuna best params: {tuned_params}")
        xgb_tuned = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=4, random_state=SEED, **tuned_params)
        xgb_tuned.fit(X_mat, y_vec - hybrid['phys_pred_train'])
        hybrid['xgb_resid'] = xgb_tuned
        hybrid['resid_pred_train'] = xgb_tuned.predict(X_mat)
        hybrid['hybrid_pred_train'] = hybrid['phys_pred_train'] + hybrid['resid_pred_train']
        hybrid['metrics_train'] = metrics_dict(y_vec, hybrid['hybrid_pred_train'])
        print_and_log(f"Updated hybrid metrics after tuning: {hybrid['metrics_train']}")
    except Exception as e:
        print_and_log(f"Optuna tuning skipped/failed: {e}")

    mlp_model = None
    scaler_mlp = None
    if TORCH_AVAILABLE:
        try:
            print_and_log("Training PyTorch MLP for MC Dropout UQ...")
            scaler_mlp = StandardScaler().fit(X_mat)
            X_mlp = scaler_mlp.transform(X_mat)
            best_mlp = optuna_mlp_tune(X_mlp, y_vec, n_trials=12) if 'optuna_mlp_tune' in globals() else {}
            mlp_params = {'hidden': (best_mlp.get('n_units1', 128), best_mlp.get('n_units2', 128)),
                          'dropout': best_mlp.get('dropout', 0.2),
                          'lr': best_mlp.get('lr', 1e-3),
                          'batch_size': best_mlp.get('batch_size', 32),
                          'n_epochs': 400, 'patience': 80}
            mlp_model = train_mlp_torch(X_mlp, y_vec, X_val=None, y_val=None, params=mlp_params)
            torch.save(mlp_model.state_dict(), os.path.join(MODEL_DIR, "mlp_dropout_final.pt"))
            joblib.dump(scaler_mlp, os.path.join(MODEL_DIR, "scaler_mlp.joblib"))
            print_and_log("Saved MLP and scaler.")
        except Exception as e:
            print_and_log(f"MLP training failed or skipped: {e}")
            mlp_model = None

    # -------- Group-wise holdout split (10% of unique base_ids) --------
    unique_ids = np.unique(groups_all)
    train_ids, holdout_ids = train_test_split(unique_ids, test_size=0.10, random_state=SEED)
    mask_holdout = np.isin(groups_all, holdout_ids)
    mask_train_full = ~mask_holdout

    X_train_full = X_mat[mask_train_full]
    y_train_full = y_vec[mask_train_full]
    X_holdout = X_mat[mask_holdout]
    y_holdout = y_vec[mask_holdout]

    scaler_phys = hybrid['scaler_phys']
    ridge = hybrid['ridge']
    xgb_resid = hybrid['xgb_resid']
    phys_indices = [FEATURE_COLS.index(c) for c in PHYSICS_COLS]

    X_holdout_phys = X_holdout[:, phys_indices]
    X_holdout_phys_scaled = scaler_phys.transform(X_holdout_phys)
    phys_holdout_pred = ridge.predict(X_holdout_phys_scaled)
    resid_holdout_pred = xgb_resid.predict(X_holdout)
    hybrid_holdout_pred = phys_holdout_pred + resid_holdout_pred
    holdout_metrics = metrics_dict(y_holdout, hybrid_holdout_pred)
    print_and_log(f"Hybrid holdout metrics (group-wise): {holdout_metrics}")
    pd.DataFrame({'meas': y_holdout, 'pred': hybrid_holdout_pred}).to_csv(os.path.join(DATA_DIR, "holdout_preds.csv"), index=False)

    preds_tree_models = []
    for name, b in baselines.items():
        try:
            preds_tree_models.append(b['model'].predict(X_holdout))
        except Exception:
            pass
    preds_tree_models = np.vstack(preds_tree_models) if len(preds_tree_models) > 0 else None
    if preds_tree_models is not None:
        ensemble_mean = preds_tree_models.mean(axis=0)
        ensemble_std = preds_tree_models.std(axis=0)
    else:
        ensemble_mean = None
        ensemble_std = None

    mlp_mean = None; mlp_std = None
    if mlp_model is not None:
        try:
            X_holdout_scaled_for_mlp = scaler_mlp.transform(X_holdout)
            mlp_mean, mlp_std = mc_dropout_predict_torch(mlp_model, X_holdout_scaled_for_mlp, n_samples=300)
        except Exception as e:
            print_and_log(f"MC Dropout failed: {e}")
            mlp_mean = None; mlp_std = None
    if ensemble_std is not None and mlp_std is not None:
        combined_unc = np.sqrt(ensemble_std**2 + mlp_std**2)
    elif ensemble_std is not None:
        combined_unc = ensemble_std
    elif mlp_std is not None:
        combined_unc = mlp_std
    else:
        combined_unc = np.full_like(hybrid_holdout_pred, np.nan)

    # -------- Group-wise conformal calibration --------
    train_group_ids = np.unique(groups_all[mask_train_full])
    train_sub_ids, cal_ids = train_test_split(train_group_ids, test_size=0.15, random_state=SEED)
    mask_cal = np.isin(groups_all, cal_ids) & mask_train_full
    X_cal = X_mat[mask_cal]
    y_cal = y_vec[mask_cal]
    phys_cal = ridge.predict(scaler_phys.transform(X_cal[:, phys_indices]))
    resid_cal_pred = xgb_resid.predict(X_cal)
    hybrid_cal_pred = phys_cal + resid_cal_pred
    nonconformity = np.abs(y_cal - hybrid_cal_pred)
    q_95 = np.quantile(nonconformity, 0.95)
    lower_cp = hybrid_holdout_pred - q_95
    upper_cp = hybrid_holdout_pred + q_95
    holdout_df = pd.DataFrame({'measured': y_holdout, 'predicted': hybrid_holdout_pred, 'lower_cp': lower_cp, 'upper_cp': upper_cp, 'unc_comb': combined_unc})
    holdout_df.to_csv(os.path.join(DATA_DIR, "holdout_with_intervals.csv"), index=False)
    try:
        np.save(os.path.join(MODEL_DIR, "q95_conformal.npy"), np.array([q_95]))
    except Exception:
        pass

    phys_coefs = dict(zip(PHYSICS_COLS, ridge.coef_.tolist()))
    print_and_log("Physics baseline coefficients:")
    print_and_log(json.dumps({'intercept': float(ridge.intercept_), 'coefs': phys_coefs}, indent=2))
    permutation_importance_and_save(xgb_resid, X_mat, y_vec - hybrid['phys_pred_train'], FEATURE_COLS, os.path.join(FIG_DIR, "perm_importance_residual.png"))
    if SHAP_AVAILABLE:
        try:
            n_sample = min(200, X_mat.shape[0])
            idx_sample = np.random.choice(X_mat.shape[0], n_sample, replace=False)
            X_sample = X_mat[idx_sample]
            compute_shap_tree_and_save(xgb_resid, X_sample, FEATURE_COLS, os.path.join(FIG_DIR, "shap_residual_summary.png"))
        except Exception as e:
            print_and_log(f"SHAP residual compute failed: {e}")

    plot_feature_corr_improved(df_aug, FEATURE_COLS, os.path.join(FIG_DIR, "fig01_feature_corr_improved.png"))

    X_scaled = StandardScaler().fit_transform(X_mat)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7,6))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_vec, cmap='plasma', s=50, edgecolor='k', linewidth=0.2)
    plt.colorbar(sc, label='Measured SFE (mJ/m²)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig02_pca_measured.png"), dpi=300)
    plt.close()

    emb, method = compute_embedding_2d(X_scaled, n_neighbors=15, min_dist=0.1, random_state=SEED)
    plt.figure(figsize=(7,6))
    sc = plt.scatter(emb[:,0], emb[:,1], c=hybrid['hybrid_pred_train'], cmap='viridis', s=40, edgecolor='k', linewidth=0.2)
    plt.colorbar(sc, label='Hybrid predicted SFE SFE (mJ/m²)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig03_embedding_predicted.png"), dpi=300)
    plt.close()

    if hybrid['cv_preds'] is not None:
        parity_plot(y_vec, hybrid['cv_preds'], os.path.join(FIG_DIR, "fig04_parity_cv.png"), title="Hybrid CV Parity (GroupKFold)")
    else:
        parity_plot(y_vec, hybrid['hybrid_pred_train'], os.path.join(FIG_DIR, "fig04_parity_train.png"), title="Hybrid Train Parity")

    residuals_plot(y_vec, hybrid['hybrid_pred_train'], os.path.join(FIG_DIR, "fig05_residuals_train.png"))
    kde_compare(y_vec, hybrid['hybrid_pred_train'], os.path.join(FIG_DIR, "fig06_kde_train.png"))

    try:
        perm_df = pd.read_csv(os.path.join(FIG_DIR, "perm_importance_residual.csv"), index_col=0, header=None, squeeze=True)
        top4 = list(perm_df.sort_values(ascending=False).index[:4])
    except Exception:
        top4 = FEATURE_COLS[:4]
    fig, axes = plt.subplots(2,2, figsize=(7,6))
    for ax, feat in zip(axes.flatten(), top4):
        try:
            PartialDependenceDisplay.from_estimator(hybrid['xgb_resid'], X_mat, [FEATURE_COLS.index(feat)], feature_names=FEATURE_COLS, ax=ax)
        except Exception:
            ax.text(0.5, 0.5, f"PDP not available for {feat}", ha='center')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(FIG_DIR, "fig08_partial_dependence.png"), dpi=300)
    plt.close()

    if TORCH_AVAILABLE and 'mlp_std' in locals() and mlp_std is not None:
        df_unc = pd.DataFrame({'pred_std': mlp_std, 'abs_err': np.abs(y_holdout - hybrid_holdout_pred)})
        df_unc['bin'] = pd.qcut(df_unc['pred_std'], q=6, duplicates='drop')
        calib = df_unc.groupby('bin').agg({'pred_std':'mean','abs_err':'mean'}).reset_index()
        plt.figure(figsize=(7,6))
        plt.plot(calib['pred_std'], calib['abs_err'], marker='o')
        plt.plot([calib['pred_std'].min(), calib['pred_std'].max()], [calib['pred_std'].min(), calib['pred_std'].max()], 'k--', label='y=x')
        plt.xlabel("Predicted std (MLP MC Dropout)")
        plt.ylabel("Mean absolute error on holdout")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig09_uncertainty_calibration.png"), dpi=300)
        plt.close()

    # -------- Fe–Ni–Cr compositional prediction surface --------
    ni_vals_pct = np.linspace(0.0, 100.0, 101)
    cr_vals_pct = np.linspace(0.0, 100.0, 101)
    pts_x, pts_y, vals_plot = [], [], []
    for ni_pct in ni_vals_pct:
        for cr_pct in cr_vals_pct:
            if ni_pct + cr_pct > 100.0:
                continue
            fe_pct = 100.0 - ni_pct - cr_pct
            frac = np.array([cr_pct/100.0, fe_pct/100.0, 0.0, ni_pct/100.0, 0.0])
            pred = get_sfe_pred_from_models(frac, FEATURE_COLS, PHYSICS_COLS, ridge=ridge, scaler_phys=scaler_phys, xgb_resid=xgb_resid)
            if not math.isnan(pred):
                pts_x.append(ni_pct)
                pts_y.append(cr_pct)
                vals_plot.append(pred)

    try:
        levels = [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 350.0]
        cmap_colors = [
            "#0033CC",
            "#66CCFF",
            "#00FFFF",
            "#CCFF99",
            "#00CC66",
            "#FFFF00",
            "#FF9900",
            "#FF0000"
        ]
        cmap = ListedColormap(cmap_colors)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        tri = Triangulation(np.asarray(pts_x), np.asarray(pts_y))
        plt.figure(figsize=(7,6))
        tcf = plt.tricontourf(tri, np.asarray(vals_plot), levels=levels, cmap=cmap, norm=norm, extend='both')
        plt.tricontour(tri, np.asarray(vals_plot), levels=levels, colors='k', linewidths=0.6)
        xs = np.linspace(0,100,200)
        plt.plot(xs, 100.0 - xs, 'k-', lw=1.0)
        cbar = plt.colorbar(tcf, ticks=levels, spacing='proportional')
        cbar.set_label('Predicted SFE (mJ/m²))')
        cbar.ax.set_yticklabels([str(int(l)) for l in levels])
        plt.xlabel("C_Ni (at.%)")
        plt.ylabel("C_Cr (at.%)")
        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        out_png = os.path.join(FIG_DIR, "fig10_pred_surface_cr_ni_pct.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved Fe-Ni-Cr contour plot (0-100 at.%): {out_png}")
    except Exception as e:
        print("Fe-Ni-Cr contour plotting failed:", e)

    # -------- Fe–Mn series: experimental vs predicted --------
    exp_points = [
        {'Fe': 84.0, 'Mn': 16.0, 'meas': 32.0},
        {'Fe': 82.0, 'Mn': 18.0, 'meas': 27.0},
        {'Fe': 80.0, 'Mn': 20.0, 'meas': 21.3},
        {'Fe': 78.0, 'Mn': 22.0, 'meas': 15.9},
        {'Fe': 75.0, 'Mn': 25.0, 'meas': 15.9},
    ]
    exp_results = []
    for p in exp_points:
        cr = 0.0; co = 0.0; ni = 0.0
        fe = p['Fe']; mn = p['Mn']
        frac = np.array([cr/100.0, fe/100.0, co/100.0, ni/100.0, mn/100.0])
        feat_dict = _compute_feature_row_from_frac(frac, FEATURE_COLS)
        X_feat_ordered = np.array([feat_dict[c] for c in FEATURE_COLS]).reshape(1, -1)
        phys_val = ridge.predict(scaler_phys.transform(np.array([feat_dict[c] for c in PHYSICS_COLS]).reshape(1,-1)))[0]
        resid_val = xgb_resid.predict(X_feat_ordered)[0]
        pred = phys_val + resid_val
        exp_results.append({'Mn_pct': p['Mn'], 'measured': p['meas'], 'predicted': float(pred)})
    df_exp = pd.DataFrame(exp_results)
    df_exp.to_csv(os.path.join(DATA_DIR, "exp_fe_mn_series_vs_pred.csv"), index=False)

    mn_vals = np.linspace(0.0, 30.0, 61)
    curve = []
    for mn in mn_vals:
        fe = 100.0 - mn
        frac = np.array([0.0, fe/100.0, 0.0, 0.0, mn/100.0])
        feat_dict = _compute_feature_row_from_frac(frac, FEATURE_COLS)
        X_feat_ordered = np.array([feat_dict[c] for c in FEATURE_COLS]).reshape(1, -1)
        phys_val = ridge.predict(scaler_phys.transform(np.array([feat_dict[c] for c in PHYSICS_COLS]).reshape(1,-1)))[0]
        resid_val = xgb_resid.predict(X_feat_ordered)[0]
        pred = phys_val + resid_val
        curve.append({'Mn_pct': mn, 'predicted': float(pred)})
    df_curve = pd.DataFrame(curve)
    df_curve.to_csv(os.path.join(DATA_DIR, "fe_mn_curve_pred.csv"), index=False)

    plt.figure(figsize=(7,6))
    plt.plot(df_curve['Mn_pct'], df_curve['predicted'], label='Hybrid predicted (Fe balance)', linewidth=2)
    plt.scatter(df_exp['Mn_pct'], df_exp['measured'], c='red', label='Experimental (user-provided)', s=60, edgecolor='k')
    plt.scatter(df_exp['Mn_pct'], df_exp['predicted'], c='blue', marker='x', label='Hybrid predicted (at exp points)', s=80)
    for _, row in df_exp.iterrows():
        plt.text(row['Mn_pct']+0.4, row['predicted']+0.4, f"{row['predicted']:.1f}", fontsize=26, color='blue')
        plt.text(row['Mn_pct']+0.4, row['measured']-0.8, f"{row['measured']:.1f}", fontsize=26, color='red')
    plt.xlabel("Mn at.% (Fe balance)")
    plt.ylabel("SFE (mJ/m²)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig11_sfe_vs_mn_exp_vs_pred.png"), dpi=300)
    plt.close()
    print(f"Saved experimental vs predicted Fe-Mn plot: {os.path.join(FIG_DIR, 'fig11_sfe_vs_mn_exp_vs_pred.png')}")

    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("SFE hybrid pipeline summary\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Original points: {len(RAW_DATA)}\n")
        f.write(f"Augmented points: {df_aug.shape[0]}\n")
        f.write("Hybrid training metrics:\n")
        f.write(json.dumps(hybrid['metrics_train'], indent=2))
        f.write("\nHybrid GroupKFold CV metrics:\n")
        f.write(json.dumps(hybrid.get('metrics_cv'), indent=2))
        f.write("\nHoldout metrics (group-wise):\n")
        f.write(json.dumps(holdout_metrics, indent=2))
        f.write("\nPhysics coefficients:\n")
        f.write(json.dumps(phys_coefs, indent=2))
    print_and_log("Saved summary and artifacts in results_sfe_full/")

    try:
        print_and_log("Training advanced models: GPR, KRR, and stacking ensemble...")
        adv = run_advanced_models(df_aug, FEATURE_COLS, PHYSICS_COLS, hybrid, baselines)
        print_and_log(f"Advanced models trained. Best: {adv.get('best')}")
    except Exception as e:
        print_and_log(f"Advanced models training failed/skipped: {e}")

    try:
        _ = generate_tetrahedral_FeNiCrMn_plot(FEATURE_COLS, PHYSICS_COLS, ridge=ridge, scaler_phys=scaler_phys, xgb_resid=xgb_resid, best_model=None, step=0.06)
    except Exception as e:
        print_and_log(f"Tetra plot generation failed: {e}")

    t1 = time.time()
    print_and_log(f"Pipeline finished in {(t1 - t0)/60.0:.2f} minutes.")



def pseudo_binary_Ni_Mn_plot(df, save_path="fig_pseudo_binary_Ni_Mn.png"):
    """
    Pseudo-binary contour for Fe–18Cr with varying Ni and Mn.
    Fe is balanced: Fe = 100 - (Cr + Ni + Mn)
    Co is fixed to 0.
    Uses trained models to predict SFE.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # -----------------------------
    # Grid definition
    # -----------------------------
    Ni = np.linspace(0.0, 20.0, 101)   # at%
    Mn = np.linspace(0.0, 20.0, 101)   # at%
    Ni_grid, Mn_grid = np.meshgrid(Ni, Mn)

    # -----------------------------
    # Fixed composition
    # -----------------------------
    Cr = 18.0
    Co = 0.0

    # Fe balance
    Fe_grid = 100.0 - (Cr + Ni_grid + Mn_grid)
    Fe_grid[Fe_grid < 0.0] = np.nan   # invalid compositions

    # -----------------------------
    # Allocate SFE array
    # -----------------------------
    SFE = np.full_like(Ni_grid, np.nan, dtype=float)

    # -----------------------------
    # Prediction loop
    # -----------------------------
    for i in range(Ni_grid.shape[0]):
        for j in range(Ni_grid.shape[1]):

            if np.isnan(Fe_grid[i, j]):
                continue

            # --- IMPORTANT ---
            # Build fraction array in canonical ELEMENTS order:
            # ['Cr', 'Fe', 'Co', 'Ni', 'Mn']
            frac = np.zeros(5, dtype=float)
            frac[0] = Cr / 100.0
            frac[1] = Fe_grid[i, j] / 100.0
            frac[2] = Co / 100.0
            frac[3] = Ni_grid[i, j] / 100.0
            frac[4] = Mn_grid[i, j] / 100.0

            try:
                SFE[i, j] = get_sfe_pred_from_models(
                    frac,
                    FEATURE_COLS,
                    PHYSICS_COLS
                )
            except Exception:
                SFE[i, j] = np.nan

    # -----------------------------
    # Safety checks
    # -----------------------------
    if np.all(np.isnan(SFE)):
        raise RuntimeError(
            "pseudo_binary_Ni_Mn_plot: All SFE predictions are NaN. "
            "Check model loading or feature construction."
        )

    # Mask invalid values for plotting
    SFE = np.ma.masked_invalid(SFE)

    # -----------------------------
    # Contour levels (robust)
    # -----------------------------
    vmin = float(SFE.min())
    vmax = float(SFE.max())

    if abs(vmax - vmin) < 1e-6:
        raise RuntimeError(
            "pseudo_binary_Ni_Mn_plot: SFE field is nearly constant; "
            "contour plot is not meaningful."
        )

    levels = np.linspace(vmin, vmax, 21)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(7, 6))

    cp = plt.contourf(
        Ni_grid,
        Mn_grid,
        SFE,
        levels=levels,
        cmap="RdBu_r"
    )

    cbar = plt.colorbar(cp)
    cbar.set_label("SFE (mJ/m²)", fontsize=26)
    cbar.ax.tick_params(labelsize=26)

    plt.xlabel("Ni (at%)", fontsize=26)
    plt.ylabel("Mn (at%)", fontsize=26)

    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("SFE pipeline script.")
        print("Examples:")
        print("  python3 sfe.py run")
        print("  python3 sfe.py predict --comp 'Fe=80 Mn=20'")
        print("  python3 sfe.py gen-fe-ni-cr --fine_step 0.01 --coarse_step 0.05")
        print("  python3 sfe.py gen-ternary --step 0.02")
        print("  python3 sfe.py plot-feat-corr")
        sys.exit(0)
    cli()
