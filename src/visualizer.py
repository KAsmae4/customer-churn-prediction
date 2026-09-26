"""
src/visualizer.py
-----------------
All EDA and model evaluation plots — clean, publication-quality.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# ─────────────────────────────────────────────
# Style Config
# ─────────────────────────────────────────────

PALETTE = {
    "primary":   "#2563EB",
    "danger":    "#DC2626",
    "success":   "#16A34A",
    "warning":   "#D97706",
    "neutral":   "#6B7280",
    "bg":        "#F8FAFC",
    "churn_yes": "#DC2626",
    "churn_no":  "#2563EB",
}

def set_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["bg"],
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linewidth":   0.6,
        "font.family":      "DejaVu Sans",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "figure.dpi":       120,
    })

set_style()

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def save(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches='tight', facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  Saved → {path.name}")
    return path


# ─────────────────────────────────────────────
# EDA Plots
# ─────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame):
    """Overall churn rate donut + bar."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Customer Churn Distribution", fontsize=15, fontweight='bold', y=1.02)

    churn_counts = df['Churn'].value_counts()
    labels = ['No Churn', 'Churn']
    colors = [PALETTE["churn_no"], PALETTE["churn_yes"]]

    # Donut
    wedges, texts, autotexts = axes[0].pie(
        churn_counts, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        textprops=dict(fontsize=11)
    )
    for at in autotexts:
        at.set_fontsize(13); at.set_fontweight('bold'); at.set_color('white')
    axes[0].set_title("Churn Breakdown", pad=10)

    # Bar
    bars = axes[1].bar(labels, churn_counts, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, churn_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    axes[1].set_title("Count of Customers")
    axes[1].set_ylabel("Count")
    axes[1].set_ylim(0, churn_counts.max() * 1.15)

    fig.tight_layout()
    return save(fig, "01_churn_distribution.png")


def plot_numerical_distributions(df: pd.DataFrame):
    """Distribution of numerical features by churn status."""
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Numerical Feature Distributions by Churn Status", fontsize=14, fontweight='bold')

    for ax, col in zip(axes, num_cols):
        for churn_val, color, label in [(0, PALETTE["churn_no"], 'No Churn'), (1, PALETTE["churn_yes"], 'Churn')]:
            data = df[df['Churn'] == churn_val][col].dropna()
            ax.hist(data, bins=35, alpha=0.55, color=color, label=label, edgecolor='white')
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()

    fig.tight_layout()
    return save(fig, "02_numerical_distributions.png")


def plot_categorical_churn(df: pd.DataFrame):
    """Churn rate by key categorical features."""
    cat_features = [
        ('Contract', 'Contract Type'),
        ('InternetService', 'Internet Service'),
        ('PaymentMethod', 'Payment Method'),
        ('tenure_group', 'Tenure Group'),
        ('SeniorCitizen', 'Senior Citizen'),
        ('Partner', 'Partner'),
    ]

    df2 = df.copy()
    df2['tenure_group'] = pd.cut(df2['tenure'], bins=[0, 12, 24, 48, 72],
                                  labels=['0-12m', '13-24m', '25-48m', '49-72m'])
    df2['SeniorCitizen'] = df2['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Churn Rate by Key Categorical Features", fontsize=15, fontweight='bold')
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, cat_features):
        try:
            churn_rate = df2.groupby(col, observed=True)['Churn'].mean().sort_values(ascending=False) * 100
            bars = ax.bar(churn_rate.index, churn_rate.values,
                          color=[PALETTE["churn_yes"] if v > 20 else PALETTE["primary"] for v in churn_rate.values],
                          edgecolor='white', linewidth=1)
            for bar, val in zip(bars, churn_rate.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_title(title)
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, min(churn_rate.max() * 1.2, 100))
            ax.tick_params(axis='x', rotation=15)
        except Exception as e:
            ax.set_title(f"{title} (error: {e})")

    fig.tight_layout()
    return save(fig, "03_categorical_churn_rates.png")


def plot_correlation_heatmap(df: pd.DataFrame, encoders: dict = None):
    """Correlation heatmap of all encoded features."""
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include='object').columns:
        from sklearn.preprocessing import LabelEncoder
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    corr = df_enc.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, linewidths=0.5,
        annot_kws={"size": 7.5}, ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight='bold', pad=15)
    fig.tight_layout()
    return save(fig, "04_correlation_heatmap.png")


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    """Horizontal bar chart of feature importances."""
    df_plot = importance_df.head(top_n).sort_values('importance')

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [PALETTE["churn_yes"] if i >= len(df_plot) - 3 else PALETTE["primary"]
              for i in range(len(df_plot))]
    bars = ax.barh(df_plot['feature'], df_plot['importance'], color=colors,
                   edgecolor='white', linewidth=1)

    for bar, val in zip(bars, df_plot['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    return save(fig, "05_feature_importance.png")


def plot_monthly_charges_vs_tenure(df: pd.DataFrame):
    """Scatter: MonthlyCharges vs Tenure colored by Churn."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for churn_val, color, label, alpha in [(0, PALETTE["churn_no"], 'No Churn', 0.3), (1, PALETTE["churn_yes"], 'Churn', 0.6)]:
        sub = df[df['Churn'] == churn_val]
        ax.scatter(sub['tenure'], sub['MonthlyCharges'], c=color, alpha=alpha,
                   label=label, s=20, linewidths=0)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Monthly Charges vs. Tenure (by Churn Status)", fontsize=13, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    return save(fig, "06_charges_vs_tenure.png")


# ─────────────────────────────────────────────
# Model Evaluation Plots
# ─────────────────────────────────────────────

def plot_confusion_matrices(results: dict):
    """Grid of confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold')

    for ax, (name, r) in zip(axes, results.items()):
        cm = r['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    linewidths=1, cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    return save(fig, "07_confusion_matrices.png")


def plot_roc_curves(results: dict, y_test: np.ndarray):
    """ROC curves for all models on the same axes."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [PALETTE["primary"], PALETTE["churn_yes"], PALETTE["success"], PALETTE["warning"]]

    for (name, r), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    return save(fig, "08_roc_curves.png")


def plot_model_comparison(summary_df: pd.DataFrame):
    """Grouped bar chart comparing model metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    models = summary_df['Model'].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    colors = [PALETTE["primary"], PALETTE["churn_yes"], PALETTE["success"], PALETTE["warning"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        row = summary_df[summary_df['Model'] == model].iloc[0]
        vals = [row[m] for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=model, color=color,
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    return save(fig, "09_model_comparison.png")


def plot_business_insights(df: pd.DataFrame):
    """3-panel business insight dashboard."""
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle("Business Insights: Churn Risk Segments", fontsize=15, fontweight='bold')

    # Panel 1: Churn by contract + tenure segment
    ax1 = fig.add_subplot(gs[0])
    df2 = df.copy()
    df2['tenure_group'] = pd.cut(df2['tenure'], bins=[0, 12, 24, 48, 72],
                                  labels=['0-12m', '13-24m', '25-48m', '49-72m'])
    pivot = df2.groupby(['Contract', 'tenure_group'], observed=True)['Churn'].mean().unstack() * 100
    pivot.plot(kind='bar', ax=ax1, colormap='RdYlBu_r', edgecolor='white', linewidth=0.8)
    ax1.set_title("Churn Rate by Contract & Tenure")
    ax1.set_ylabel("Churn Rate (%)")
    ax1.set_xlabel("Contract Type")
    ax1.tick_params(axis='x', rotation=15)
    ax1.legend(title='Tenure', fontsize=8)

    # Panel 2: Avg Monthly Charges for churned vs not
    ax2 = fig.add_subplot(gs[1])
    charge_data = df.groupby(['InternetService', 'Churn'])['MonthlyCharges'].mean().unstack()
    charge_data.columns = ['No Churn', 'Churn']
    charge_data.plot(kind='bar', ax=ax2,
                     color=[PALETTE["churn_no"], PALETTE["churn_yes"]],
                     edgecolor='white', linewidth=0.8)
    ax2.set_title("Avg Monthly Charges\nby Internet Service")
    ax2.set_ylabel("Avg Monthly Charges ($)")
    ax2.set_xlabel("Internet Service")
    ax2.tick_params(axis='x', rotation=0)

    # Panel 3: Payment method churn
    ax3 = fig.add_subplot(gs[2])
    pm_churn = df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False) * 100
    short_labels = [p.replace(' (automatic)', '\n(auto)').replace('Electronic check', 'E-Check')
                    .replace('Mailed check', 'Mail Check')
                    for p in pm_churn.index]
    bars = ax3.bar(short_labels, pm_churn.values,
                   color=[PALETTE["churn_yes"] if v > 25 else PALETTE["primary"] for v in pm_churn.values],
                   edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, pm_churn.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax3.set_title("Churn Rate by Payment Method")
    ax3.set_ylabel("Churn Rate (%)")
    ax3.tick_params(axis='x', rotation=15)

    fig.tight_layout()
    return save(fig, "10_business_insights.png")
