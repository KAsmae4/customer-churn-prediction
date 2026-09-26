"""
src/model_trainer.py
---------------------
Train, evaluate, and persist ML models for churn prediction.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────

def get_models():
    """Return dict of model instances."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=4,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ),
    }

    # Try XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1
        )
        print("[models] XGBoost available ✓")
    except ImportError:
        print("[models] XGBoost not available, using Gradient Boosting instead")

    return models


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute full evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def cross_validate_model(model, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> dict:
    """Run stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    return {
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
        "cv_scores": scores
    }


def train_all_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Train all models, evaluate, cross-validate.
    Returns results dict keyed by model name.
    """
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  Training: {name}")
        print(f"{'─'*50}")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        cv_metrics = cross_validate_model(model, X_train, y_train)

        results[name] = {
            "model": model,
            **metrics,
            **cv_metrics,
        }

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  CV AUC   : {cv_metrics['cv_mean']:.4f} ± {cv_metrics['cv_std']:.4f}")

    return results


def select_best_model(results: dict) -> tuple:
    """Select best model by ROC-AUC score."""
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best = results[best_name]
    print(f"\n✅ Best Model: {best_name}  (ROC-AUC: {best['roc_auc']:.4f})")
    return best_name, best['model']


def save_model(model, name: str = "best_model"):
    """Persist model to disk."""
    path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, path)
    print(f"[save] Model saved → {path}")
    return path


def load_model(name: str = "best_model"):
    """Load persisted model."""
    path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
    return joblib.load(path)


def get_summary_table(results: dict) -> pd.DataFrame:
    """Build a clean summary DataFrame of all model metrics."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1-Score": round(r["f1"], 4),
            "ROC-AUC": round(r["roc_auc"], 4),
            "CV AUC (mean)": round(r["cv_mean"], 4),
            "CV AUC (±std)": round(r["cv_std"], 4),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw_data, clean_data
    from feature_engineering import prepare_data

    df = load_raw_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test, features, encoders, scaler, importance_df = prepare_data(df)

    results = train_all_models(X_train, X_test, y_train, y_test)
    best_name, best_model = select_best_model(results)
    save_model(best_model, best_name)

    print("\n📊 Summary Table:")
    print(get_summary_table(results).to_string(index=False))
