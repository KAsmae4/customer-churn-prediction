"""
app/streamlit_app.py
---------------------
Streamlit-based churn prediction web application.
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor | Telco Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #F8FAFC; }
  .stMetric { background: white; border-radius: 10px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .churn-yes { background: #FEE2E2; border-left: 5px solid #DC2626; border-radius: 8px; padding: 16px; }
  .churn-no  { background: #DCFCE7; border-left: 5px solid #16A34A; border-radius: 8px; padding: 16px; }
  .insight-card { background: white; border-radius: 10px; padding: 16px; margin: 8px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  h1 { color: #1E3A5F; }
  h2, h3 { color: #2563EB; }
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        encoders   = joblib.load(MODELS_DIR / "encoders.pkl")
        scaler     = joblib.load(MODELS_DIR / "scaler.pkl")
        features   = joblib.load(MODELS_DIR / "feature_names.pkl")

        with open(MODELS_DIR / "best_model_name.json") as f:
            best_name = json.load(f)["name"]

        model_file = best_name.replace(' ', '_').lower() + ".pkl"
        model = joblib.load(MODELS_DIR / model_file)
        return model, encoders, scaler, features, best_name
    except FileNotFoundError as e:
        return None, None, None, None, None


model, encoders, scaler, feature_names, model_name = load_artifacts()


# ── Predict helper ─────────────────────────────────────────────
def predict_churn(input_dict: dict) -> tuple:
    """Encode, scale, predict. Returns (label, probability)."""
    from sklearn.preprocessing import LabelEncoder

    df_input = pd.DataFrame([input_dict])

    # Encode categoricals
    for col, le in encoders.items():
        if col in df_input.columns:
            val = df_input[col].astype(str).values[0]
            if val in le.classes_:
                df_input[col] = le.transform([val])
            else:
                df_input[col] = le.transform([le.classes_[0]])

    # Select and order features
    for f in feature_names:
        if f not in df_input.columns:
            df_input[f] = 0
    df_input = df_input[feature_names]

    X_sc = scaler.transform(df_input.values)
    prob = model.predict_proba(X_sc)[0][1]
    label = "Churn" if prob >= 0.5 else "No Churn"
    return label, prob


def gauge_chart(prob: float):
    """Simple matplotlib gauge chart for churn probability."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    color = '#DC2626' if prob > 0.6 else ('#D97706' if prob > 0.35 else '#16A34A')
    theta = np.pi * (1 - prob)
    ax.add_patch(plt.matplotlib.patches.Wedge((0.5, 0), 0.42, 0, 180, color='#E5E7EB'))
    ax.add_patch(plt.matplotlib.patches.Wedge((0.5, 0), 0.42, 0, 180 * prob, color=color, alpha=0.85))
    ax.text(0.5, 0.15, f"{prob:.0%}", ha='center', va='center',
            fontsize=22, fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.0, "Churn Probability", ha='center', va='center',
            fontsize=9, color='#6B7280', transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 0.55)
    ax.axis('off')
    fig.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — NAVIGATION
# ══════════════════════════════════════════════════════════════════

st.sidebar.image("https://img.icons8.com/color/96/000000/signal.png", width=60)
st.sidebar.title("📡 Churn Predictor")
st.sidebar.caption("Telco Customer Intelligence Platform")
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict Churn", "📊 EDA Dashboard", "🤖 Model Insights", "💡 Business Strategy"]
)

if model:
    st.sidebar.success(f"✅ Model loaded: **{model_name}**")
else:
    st.sidebar.error("⚠️ Models not trained. Run `src/train_pipeline.py` first.")


# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════════

if page == "🔮 Predict Churn":
    st.title("🔮 Customer Churn Prediction")
    st.caption("Enter customer details below to predict churn probability in real-time.")

    if not model:
        st.warning("⚠️ Please run the training pipeline first: `python src/train_pipeline.py`")
        st.stop()

    # ── Input form ─────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("👤 Customer Profile")
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                             float(round(tenure * monthly_charges, 2)))

        with c3:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.subheader("📶 Services")
        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        with s2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with s3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    if submitted:
        input_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }

        label, prob = predict_churn(input_data)
        risk = "🔴 HIGH RISK" if prob > 0.6 else ("🟡 MEDIUM RISK" if prob > 0.35 else "🟢 LOW RISK")

        st.markdown("---")
        col_gauge, col_result = st.columns([1, 2])

        with col_gauge:
            st.pyplot(gauge_chart(prob), use_container_width=True)

        with col_result:
            if label == "Churn":
                st.markdown(f"""
                <div class="churn-yes">
                <h2>⚠️ {risk}</h2>
                <h3>Prediction: <strong>WILL CHURN</strong></h3>
                <p>Churn Probability: <strong>{prob:.1%}</strong></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="churn-no">
                <h2>✅ {risk}</h2>
                <h3>Prediction: <strong>WILL STAY</strong></h3>
                <p>Churn Probability: <strong>{prob:.1%}</strong></p>
                </div>""", unsafe_allow_html=True)

            st.markdown("#### Key Risk Factors for This Customer:")
            risks = []
            if contract == "Month-to-month": risks.append("📋 Month-to-month contract (no lock-in)")
            if internet == "Fiber optic": risks.append("📶 Fiber optic — higher cost, higher churn risk")
            if payment == "Electronic check": risks.append("💳 Electronic check payment method")
            if tenure < 12: risks.append(f"⏱️ Low tenure ({tenure}m) — high early churn risk")
            if senior == "Yes": risks.append("👴 Senior citizen — price sensitive segment")
            if tech_support == "No": risks.append("🔧 No tech support subscription")

            if risks:
                for r in risks:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- ✅ No major risk factors detected")


# ══════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════

elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")

    img_map = {
        "01_churn_distribution.png":   "Churn Distribution",
        "02_numerical_distributions.png": "Numerical Feature Distributions",
        "03_categorical_churn_rates.png": "Churn Rate by Categorical Features",
        "04_correlation_heatmap.png":  "Feature Correlation Heatmap",
        "05_feature_importance.png":   "Feature Importances",
        "06_charges_vs_tenure.png":    "Monthly Charges vs Tenure",
        "10_business_insights.png":    "Business Insight Dashboard",
    }

    for filename, title in img_map.items():
        path = OUTPUTS_DIR / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_column_width=True)
            st.markdown("---")
        else:
            st.info(f"⏳ {title} — run `train_pipeline.py` to generate")


# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════

elif page == "🤖 Model Insights":
    st.title("🤖 Model Performance & Evaluation")

    if not model:
        st.warning("⚠️ Run `python src/train_pipeline.py` first.")
    else:
        st.success(f"Best model: **{model_name}**")

    for filename, title in [
        ("07_confusion_matrices.png", "Confusion Matrices"),
        ("08_roc_curves.png",         "ROC Curves"),
        ("09_model_comparison.png",   "Model Comparison"),
    ]:
        path = OUTPUTS_DIR / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_column_width=True)
            st.markdown("---")
        else:
            st.info(f"⏳ {title} — run `train_pipeline.py` to generate")


# ══════════════════════════════════════════════════════════════════
#  PAGE 4 — BUSINESS STRATEGY
# ══════════════════════════════════════════════════════════════════

elif page == "💡 Business Strategy":
    st.title("💡 Business Insights & Retention Strategies")
    st.caption("Data-driven retention playbook for reducing customer churn.")

    strategies = [
        {
            "impact": "🔴 HIGH",
            "segment": "Month-to-Month + New Customers (0-12m)",
            "driver": "No lock-in, low switching cost, early dissatisfaction",
            "action": "Offer 3-month free upgrade or loyalty discount after month 6 to convert to annual plan.",
            "kpi": "Contract upgrade rate, Month-6 NPS"
        },
        {
            "impact": "🔴 HIGH",
            "segment": "Fiber Optic + Electronic Check Payers",
            "driver": "High bills ($65-90/mo) combined with payment friction",
            "action": "Auto-pay enrollment campaign with 5% discount. Proactive billing reviews.",
            "kpi": "Auto-pay conversion rate, ARPU retention"
        },
        {
            "impact": "🟡 MEDIUM",
            "segment": "No Tech Support / No Online Security",
            "driver": "Perceived low value; technical issues go unresolved",
            "action": "Bundle tech support + security at $5/mo (vs $10 individually).",
            "kpi": "Add-on attach rate, CSAT"
        },
        {
            "impact": "🟡 MEDIUM",
            "segment": "Senior Citizens on Month-to-Month",
            "driver": "Fixed-income price sensitivity, less digital adoption",
            "action": "Senior loyalty programme with fixed-price guarantee (2 years) + dedicated phone line.",
            "kpi": "Senior retention rate, CSAT"
        },
        {
            "impact": "🟢 LOW",
            "segment": "High Tenure (>36m) Risk",
            "driver": "Long-term customers feel undervalued vs. new-customer promotions",
            "action": "VIP loyalty perks: free equipment upgrades, priority support, anniversary discounts.",
            "kpi": "Loyalty programme enrolment, NPS 3yr+ cohort"
        },
    ]

    for s in strategies:
        with st.expander(f"{s['impact']} — {s['segment']}", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**🔍 Churn Driver**\n\n{s['driver']}")
            with c2:
                st.markdown(f"**🎯 Retention Action**\n\n{s['action']}")
            st.markdown(f"**📈 KPIs to Track:** `{s['kpi']}`")

    st.markdown("---")
    st.subheader("📌 Executive Summary")
    st.markdown("""
    | Insight | Finding |
    |---------|---------|
    | 🏆 #1 Churn Driver | Month-to-month contract (~43% churn rate) |
    | 📶 Service Risk | Fiber Optic customers churn 2× more than DSL |
    | 💳 Payment Risk | Electronic check payers churn 30%+ vs 15% auto-pay |
    | ⏱️ Tenure Risk | First 12 months are critical — churn peaks early |
    | 👴 Segment Risk | Senior citizens with no support plans = highest risk |
    | 💰 Revenue Impact | Each 1% churn reduction ≈ $300K+ annual revenue saved |
    """)
