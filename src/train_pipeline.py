"""
src/train_pipeline.py
----------------------
End-to-end training pipeline: EDA → Feature Engineering → Modelling → Insights.
Run this script to train all models and generate all visualizations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_raw_data, clean_data
from feature_engineering import prepare_data
from model_trainer import (
    train_all_models, select_best_model,
    save_model, get_summary_table
)
from visualizer import (
    plot_churn_distribution,
    plot_numerical_distributions,
    plot_categorical_churn,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_monthly_charges_vs_tenure,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison,
    plot_business_insights,
)
from business_insights import print_insights
import joblib


def run():
    print("\n" + "█" * 60)
    print("  CUSTOMER CHURN ANALYSIS & PREDICTION PIPELINE")
    print("█" * 60)

    # ── 1. Load & Clean ──────────────────────────────────────────
    print("\n[1/6] Loading and cleaning data...")
    df = load_raw_data()
    df = clean_data(df)

    # ── 2. EDA Visualizations ────────────────────────────────────
    print("\n[2/6] Generating EDA visualizations...")
    plot_churn_distribution(df)
    plot_numerical_distributions(df)
    plot_categorical_churn(df)
    plot_correlation_heatmap(df)
    plot_monthly_charges_vs_tenure(df)
    plot_business_insights(df)

    # ── 3. Feature Engineering ───────────────────────────────────
    print("\n[3/6] Feature engineering...")
    X_train, X_test, y_train, y_test, features, encoders, scaler, importance_df = prepare_data(df)
    plot_feature_importance(importance_df)

    # ── 4. Model Training ────────────────────────────────────────
    print("\n[4/6] Training models...")
    results = train_all_models(X_train, X_test, y_train, y_test)
    best_name, best_model = select_best_model(results)
    save_model(best_model, best_name)

    # Save metadata for app
    import json
    models_dir = Path(__file__).parent.parent / "models"
    with open(models_dir / "best_model_name.json", "w") as f:
        json.dump({"name": best_name}, f)

    # ── 5. Evaluation Plots ──────────────────────────────────────
    print("\n[5/6] Generating evaluation plots...")
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    summary_df = get_summary_table(results)
    plot_model_comparison(summary_df)

    # ── 6. Business Insights ─────────────────────────────────────
    print("\n[6/6] Business insights...")
    print_insights(df)

    # Summary table
    print("\n📊 MODEL PERFORMANCE SUMMARY:")
    print(summary_df.to_string(index=False))

    best_metrics = results[best_name]
    print(f"\n✅ Best Model  : {best_name}")
    print(f"   Accuracy   : {best_metrics['accuracy']:.4f}")
    print(f"   F1-Score   : {best_metrics['f1']:.4f}")
    print(f"   ROC-AUC    : {best_metrics['roc_auc']:.4f}")
    print(f"   CV AUC     : {best_metrics['cv_mean']:.4f} ± {best_metrics['cv_std']:.4f}")
    print("\n✅ Pipeline complete! All outputs saved to /outputs/")
    print("   Run the app with: streamlit run app/streamlit_app.py")


if __name__ == '__main__':
    run()
