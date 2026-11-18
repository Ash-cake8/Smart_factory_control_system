"""
common.py - shared data layer for Stage 1 (binary) and Stage 2 (5-class).

Everything the two stages have in common lives here so that the whole team
uses exactly the same cleaning, features, split and preprocessing:

    df            = load_clean()               # cleaned dataframe (10,000 rows)
    train, val, test = make_splits(df)          # ONE stratified 70/15/15 split
    X, y          = stage1_xy(train)            # Stage 1: failure vs no failure
    X, y          = stage2_xy(train)            # Stage 2: failure rows only, 5 classes
    prep          = build_preprocessor()        # unfitted ColumnTransformer
    pipe          = make_pipeline(prep, <any estimator>)   # leakage-safe in CV (used by the team)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "predictive_maintenance.csv"
CLEAN_REF_PATH = ROOT / "data" / "raw" / "predictive_maintenance_clean.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
for _d in (PROCESSED_DIR, ARTIFACTS_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NO_FAILURE = "No Failure"
# alphabetical == sklearn's class order
FAILURE_CLASSES = [
    "Heat Dissipation Failure",
    "Overstrain Failure",
    "Power Failure",
    "Random Failures",
    "Tool Wear Failure",
]

CATEGORICAL_FEATURES = ["Type"]  # product quality tier: L < M < H
NUMERIC_FEATURES = [
    "Air temperature [C]",
    "Process temperature [C]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_diff_C",
    "Power_W",
    "Wear_x_Torque",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
STAGE1_TARGET = "Target"  # 0 = no failure, 1 = failure
STAGE2_TARGET = "Failure Type"  # 5 failure modes (failure rows only)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw AI4I CSV -> modelling dataframe. Deterministic, no fitting involved."""
    df = raw.copy()

    # 1) identifiers carry no physical signal (Product ID = type letter + serial)
    df = df.drop(columns=["UDI", "Product ID"])

    # 2) categorical hygiene
    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    assert set(df["Type"]) <= {"L", "M", "H"}, "unexpected Type value"
    df["Failure Type"] = df["Failure Type"].astype(str).str.strip()

    # 3) label consistency: `Failure Type` is the source of truth.
    #    Raw file has 18 'Random Failures' rows with Target=0 and
    #    9 rows with Target=1 but 'No Failure' -> 27 contradictory rows.
    df["Target"] = (df["Failure Type"] != NO_FAILURE).astype(int)

    # 4) units: Kelvin -> Celsius (same convention as predictive_maintenance_clean.csv)
    df["Air temperature [C]"] = (df.pop("Air temperature [K]") - 273.15).round(2)
    df["Process temperature [C]"] = (df.pop("Process temperature [K]") - 273.15).round(2)

    # 5) physics-based features (functions of the inputs only -> no label leakage)
    df["Temp_diff_C"] = (df["Process temperature [C]"] - df["Air temperature [C]"]).round(2)
    df["Power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    df["Wear_x_Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]  # overstrain driver

    return df[FEATURES + [STAGE1_TARGET, STAGE2_TARGET]]


def load_clean() -> pd.DataFrame:
    return clean_raw(pd.read_csv(RAW_PATH))


# --------------------------------------------------------------------------- #
# Split (one split shared by both stages)
# --------------------------------------------------------------------------- #
def make_splits(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=RANDOM_STATE):
    """Stratified 70/15/15 split on the 6-class `Failure Type`.

    Stratifying on the finest label keeps BOTH the binary ratio (Stage 1) and the
    5 failure-mode ratios (Stage 2) intact in every split. Stage-2 rows of a
    split are exactly the failure rows of the same Stage-1 split, so there is
    no leakage between stages. Original row index is preserved.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[STAGE2_TARGET], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[STAGE2_TARGET],
        random_state=seed,
    )
    return train.sort_index(), val.sort_index(), test.sort_index()


def stage1_xy(split: pd.DataFrame):
    """Stage 1: all rows, binary target."""
    return split[FEATURES], split[STAGE1_TARGET]


def stage2_xy(split: pd.DataFrame):
    """Stage 2: failure rows only, 5-class target."""
    f = split[split[STAGE1_TARGET] == 1]
    return f[FEATURES], f[STAGE2_TARGET]


# --------------------------------------------------------------------------- #
# Shared preprocessing (encoding + scaling)
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Unfitted. Put it INSIDE a Pipeline so CV re-fits it per fold (no leakage).

    - Type  -> ordinal (L=0, M=1, H=2; it is an ordered quality tier) -> scaled
    - numeric features -> StandardScaler
    (scaling is irrelevant for trees but harmless, and required for Logistic Regression)
    """
    type_branch = Pipeline(
        [
            ("ord", OrdinalEncoder(categories=[["L", "M", "H"]],
                                   handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scale", StandardScaler()),
        ]
    )
    prep = ColumnTransformer(
        [("type", type_branch, CATEGORICAL_FEATURES),
         ("num", StandardScaler(), NUMERIC_FEATURES)],
        verbose_feature_names_out=False,
    )
    return prep.set_output(transform="pandas")
