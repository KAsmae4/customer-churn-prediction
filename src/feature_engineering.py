"""
src/feature_engineering.py
---------------------------
Feature encoding, scaling, and selection pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def encode_features(df: pd.DataFrame):
    """
    Encode all categorical columns with LabelEncoder.
    Returns encoded DataFrame and dict of encoders.
    """
    df = df.copy()
    encoders = {}
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    print(f"[encode] Encoded {len(cat_cols)} categorical columns: {cat_cols}")
    return df, encoders


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Standard scale numeric features. Returns scaled arrays + fitted scaler."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


def select_features(X: pd.DataFrame, y: pd.Series, n_features: int = 15):
    """
    Use Random Forest feature importance to select top-N features.
    Returns selected column names + importance DataFrame.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    top_features = importance_df['feature'].head(n_features).tolist()
    print(f"[select] Top {n_features} features selected")
    return top_features, importance_df


def handle_imbalance(X: np.ndarray, y: np.ndarray, method: str = 'oversample'):
    """
    Handle class imbalance.
    method='oversample': manual random oversampling (no external lib needed)
    method='smote': use SMOTE if imbalanced-learn is available
    """
    from collections import Counter
    counts = Counter(y)
    print(f"[imbalance] Class distribution before: {dict(counts)}")

    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            print(f"[imbalance] SMOTE applied: {Counter(y_res)}")
            return X_res, y_res
        except ImportError:
            print("[imbalance] imblearn not available, falling back to random oversample")

    # Manual random oversampling
    majority_class = max(counts, key=counts.get)
    minority_class = min(counts, key=counts.get)
    n_majority = counts[majority_class]
    n_minority = counts[minority_class]

    minority_idx = np.where(y == minority_class)[0]
    oversample_idx = np.random.choice(minority_idx, n_majority - n_minority, replace=True)

    X_res = np.vstack([X, X[oversample_idx]])
    y_res = np.concatenate([y, y[oversample_idx]])

    print(f"[imbalance] Oversampled: {Counter(y_res)}")
    return X_res, y_res


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, balance: bool = True):
    """
    Full pipeline: encode → split → scale → (balance train) → select features.
    Returns (X_train, X_test, y_train, y_test, feature_names, encoders, scaler, importance_df)
    """
    target = 'Churn'
    df_enc, encoders = encode_features(df)

    X = df_enc.drop(columns=[target])
    y = df_enc[target]

    # Feature selection on full dataset
    top_features, importance_df = select_features(X, y)
    X = X[top_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train_sc, X_test_sc, scaler = scale_features(X_train.values, X_test.values)

    if balance:
        X_train_sc, y_train_bal = handle_imbalance(X_train_sc, y_train.values)
    else:
        y_train_bal = y_train.values

    # Save artifacts
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(top_features, MODELS_DIR / "feature_names.pkl")

    print(f"[prepare] Train size: {X_train_sc.shape}, Test size: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train_bal, y_test.values, top_features, encoders, scaler, importance_df


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw_data, clean_data

    df = load_raw_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test, features, *_ = prepare_data(df)
    print("Features:", features)
    print("Train shape:", X_train.shape)
