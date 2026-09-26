"""
src/data_loader.py
------------------
Data loading, cleaning, and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw_data(path: str = None) -> pd.DataFrame:
    """Load raw Telco churn dataset."""
    filepath = path or RAW_DATA_PATH
    df = pd.read_csv(filepath)
    print(f"[load] Shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Drop customerID
    - Convert TotalCharges to numeric
    - Fill missing TotalCharges with tenure * MonthlyCharges
    - Encode Churn to binary
    """
    df = df.copy()

    # Drop ID
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    # TotalCharges: coerce spaces / empty strings to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing TotalCharges (new customers with 0 tenure)
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'tenure'] * df.loc[mask, 'MonthlyCharges']
    print(f"[clean] Filled {mask.sum()} missing TotalCharges values")

    # Binary target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    print(f"[clean] Shape after cleaning: {df.shape}")
    print(f"[clean] Churn rate: {df['Churn'].mean():.1%}")
    return df


def get_feature_types(df: pd.DataFrame):
    """Return lists of categorical and numerical columns (excl. target)."""
    target = 'Churn'
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    return cat_cols, num_cols


if __name__ == '__main__':
    df = load_raw_data()
    df = clean_data(df)
    print(df.dtypes)
    print(df.head(3))
