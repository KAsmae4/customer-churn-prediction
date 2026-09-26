# 📡 Customer Churn Analysis & Prediction System

> **Production-ready ML pipeline for predicting Telco customer churn with an interactive Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)

---

## 🎯 Project Overview

This project performs **end-to-end customer churn analysis** on the Telco Customer Churn dataset (~7,000 records). It covers:

- Exploratory Data Analysis with 10 professional visualizations
- Feature engineering with importance selection & class imbalance handling
- Multi-model ML training (Logistic Regression, Random Forest, Gradient Boosting/XGBoost)
- Business insights with actionable retention strategies
- Production-ready Streamlit web app for real-time predictions

---

## 📁 Project Structure

```
churn_analysis/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│   └── generate_dataset.py                     # Synthetic data generator
│
├── notebooks/
│   └── churn_analysis_eda.ipynb               # Full EDA notebook
│
├── src/
│   ├── data_loader.py         # Data loading & cleaning
│   ├── feature_engineering.py # Encoding, scaling, selection, balancing
│   ├── model_trainer.py       # Model training & evaluation
│   ├── visualizer.py          # All EDA & evaluation plots
│   ├── business_insights.py   # Risk segments & retention strategies
│   └── train_pipeline.py      # 🚀 Master training script
│
├── models/                    # Saved model artifacts (auto-generated)
│   ├── best_model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── app/
│   └── streamlit_app.py       # Interactive web application
│
├── outputs/                   # Generated charts (auto-generated)
│   ├── 01_churn_distribution.png
│   ├── 02_numerical_distributions.png
│   ├── 03_categorical_churn_rates.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_feature_importance.png
│   ├── 06_charges_vs_tenure.png
│   ├── 07_confusion_matrices.png
│   ├── 08_roc_curves.png
│   ├── 09_model_comparison.png
│   └── 10_business_insights.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/yourname/churn-analysis.git
cd churn-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate / Place Dataset

```bash
# Option A: Generate synthetic dataset (mirrors Kaggle Telco Churn structure)
cd data && python generate_dataset.py

# Option B: Download real dataset from Kaggle
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place as: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 3. Run the Training Pipeline

```bash
# From project root — trains all models and generates all visualizations
python src/train_pipeline.py
```

This will:
- Clean and preprocess the data
- Generate 10 EDA/evaluation visualizations in `/outputs/`
- Train Logistic Regression, Random Forest, and Gradient Boosting
- Save the best model and preprocessing artifacts to `/models/`
- Print business insights to the console

### 4. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### 5. Explore the Jupyter Notebook

```bash
jupyter notebook notebooks/churn_analysis_eda.ipynb
```

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Gradient Boosting | 0.676 | 0.313 | 0.702 | 0.433 | **0.749** |
| Logistic Regression | 0.632 | 0.294 | 0.774 | 0.426 | 0.744 |
| Random Forest | 0.703 | 0.325 | 0.641 | 0.432 | 0.744 |

> **Note:** Models are tuned for **high recall** (catch as many churners as possible). ROC-AUC is the primary ranking metric. CV AUC (5-fold) for Gradient Boosting: **0.843 ± 0.006**

---

## 🔍 Key Findings

### Top Churn Drivers
1. **Contract Type** — Month-to-month customers churn at ~37% vs ~3% for two-year contracts
2. **Early Tenure** — First 12 months are the highest-risk window (churn >33%)
3. **Fiber Optic + High Charges** — Fiber customers pay more but churn 2× more than DSL
4. **Electronic Check Payment** — Correlated with ~30%+ churn vs ~15% for auto-pay methods
5. **Senior Citizens** — Without tech support, churn significantly more

### Top 3 Risk Segments
| Rank | Segment | Churn Rate |
|------|---------|-----------|
| 🥇 | Month-to-month, Paperless Billing | ~37% |
| 🥈 | Month-to-month, 0-12m tenure | ~34% |
| 🥉 | Month-to-month, Fiber Optic | ~33% |

### Retention Strategies
| Priority | Segment | Action |
|----------|---------|--------|
| 🔴 HIGH | New month-to-month customers | 6-month loyalty discount → push annual contract |
| 🔴 HIGH | Fiber + E-check payers | Auto-pay 5% discount campaign |
| 🟡 MEDIUM | No tech support subscribers | Bundle security + support at $5/mo |
| 🟡 MEDIUM | Senior citizens (M2M) | Fixed-price senior programme (2yr guarantee) |

---

## 🖥️ Streamlit App Features

The app has 4 pages:

| Page | Description |
|------|-------------|
| 🔮 **Predict Churn** | Enter customer details → get churn probability + risk factors |
| 📊 **EDA Dashboard** | View all 10 visualizations |
| 🤖 **Model Insights** | Confusion matrices, ROC curves, model comparison |
| 💡 **Business Strategy** | Risk segments, retention playbook, executive summary |

---

## ☁️ Deployment

### Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → select `app/streamlit_app.py` as entry point
4. Add `requirements.txt` to repo root (already included)
5. **Important:** Run `train_pipeline.py` locally first, then commit the `models/` and `outputs/` directories

### Deploy to Render

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect GitHub repo
3. Set **Build Command:** `pip install -r requirements.txt && python src/train_pipeline.py`
4. Set **Start Command:** `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Data manipulation | Pandas, NumPy |
| ML models | scikit-learn, XGBoost (optional) |
| Class balancing | imbalanced-learn / manual oversampling |
| Visualizations | Matplotlib, Seaborn |
| Model persistence | joblib |
| Web app | Streamlit |
| Notebook | Jupyter |

---

## 📈 Future Improvements

- [ ] Hyperparameter tuning with Optuna
- [ ] SHAP explanations per customer
- [ ] Customer Lifetime Value (CLV) feature engineering
- [ ] Real-time model monitoring with Evidently AI
- [ ] REST API with FastAPI for microservice deployment
- [ ] Docker containerization

