"""Data loader and preprocessing module for Heart Disease prediction."""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    ALL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    path = filepath or RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    logger.info("Loaded raw dataset with shape %s from %s", df.shape, path)
    return df


def clean_and_validate_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Clean the dataset, handle medical anomalies/missing sentinel values,
    and validate features and schema.
    
    In the Cleveland dataset:
    - `ca`: values are 0-3; value 4 represents missing fluoroscopy (5 occurrences).
    - `thal`: values are 1 (normal), 2 (fixed defect), 3 (reversible defect); 
              value 0 is an undocumented missing code (2 occurrences).
    We handle these by setting them to NaN and imputing with the median/mode.
    """
    cleaned = df.copy()

    # Drop duplicate records to prevent data leakage across splits
    if drop_duplicates:
        initial_len = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        dropped = initial_len - len(cleaned)
        if dropped > 0:
            logger.info("Dropped %d duplicate row(s)", dropped)

    # Validate target presence and normalize standard medical definition:
    # 1 = Heart Disease Present (High Risk / Positive), 0 = Normal / Healthy (Negative)
    if TARGET in cleaned.columns:
        if cleaned[TARGET].isnull().any():
            missing_target_count = cleaned[TARGET].isnull().sum()
            logger.warning("Dropping %d rows with missing target values", missing_target_count)
            cleaned = cleaned.dropna(subset=[TARGET]).reset_index(drop=True)
        # Check if the raw dataset had inverted target (where oldpeak/ca were negatively correlated with target)
        # In authentic clinical data, severe ST depression (oldpeak) and vessel narrowing (ca) correlate positively with Heart Disease
        corr_oldpeak = cleaned["oldpeak"].corr(cleaned[TARGET])
        if corr_oldpeak < 0:
            logger.info("Aligning target encoding: Inverting so 1 = Heart Disease Present, 0 = Healthy/Normal")
            cleaned[TARGET] = 1 - cleaned[TARGET]

    # Handle domain-specific sentinel codes in raw Cleveland data
    if "ca" in cleaned.columns:
        # ca == 4 is sentinel for missing vessel data
        mask_ca = cleaned["ca"] == 4
        if mask_ca.any():
            logger.info("Found %d rows with ca=4 (sentinel missing); imputing with mode", mask_ca.sum())
            mode_ca = cleaned.loc[~mask_ca, "ca"].mode()[0]
            cleaned.loc[mask_ca, "ca"] = mode_ca

    if "thal" in cleaned.columns:
        # thal == 0 is sentinel for missing defect data
        mask_thal = cleaned["thal"] == 0
        if mask_thal.any():
            logger.info("Found %d rows with thal=0 (sentinel missing); imputing with mode", mask_thal.sum())
            mode_thal = cleaned.loc[~mask_thal, "thal"].mode()[0]
            cleaned.loc[mask_thal, "thal"] = mode_thal

    # Ensure correct data types
    for col in NUMERICAL_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype(int)

    logger.info("Data cleaning completed. Cleaned shape: %s", cleaned.shape)
    return cleaned


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets with stratification.
    Strict featurization rule: Split BEFORE any preprocessing or scaling.
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataframe.")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy().astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split dataset: Train shape: %s (pos: %d, neg: %d), Test shape: %s (pos: %d, neg: %d)",
        X_train.shape,
        (y_train == 1).sum(),
        (y_train == 0).sum(),
        X_test.shape,
        (y_test == 1).sum(),
        (y_test == 0).sum(),
    )
    return X_train, X_test, y_train, y_test
