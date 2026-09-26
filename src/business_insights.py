"""
src/business_insights.py
-------------------------
Identify churn risk segments and generate actionable recommendations.
"""

import pandas as pd
import numpy as np


def get_churn_risk_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify top churn-risk segments using multi-factor analysis.
    Returns a DataFrame with segment name, churn rate, and customer count.
    """
    df2 = df.copy()
    df2['tenure_group'] = pd.cut(
        df2['tenure'], bins=[0, 12, 24, 48, 72],
        labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months']
    )
    df2['charge_tier'] = pd.cut(
        df2['MonthlyCharges'], bins=[0, 35, 65, 200],
        labels=['Low (<$35)', 'Mid ($35-65)', 'High (>$65)']
    )

    segments = []

    combos = [
        ('Contract', 'tenure_group'),
        ('Contract', 'InternetService'),
        ('PaymentMethod', 'InternetService'),
        ('tenure_group', 'charge_tier'),
        ('SeniorCitizen', 'Contract'),
    ]

    for col1, col2 in combos:
        grp = df2.groupby([col1, col2], observed=True).agg(
            ChurnRate=('Churn', 'mean'),
            Count=('Churn', 'count')
        ).reset_index()
        grp['Segment'] = grp[col1].astype(str) + ' + ' + grp[col2].astype(str)
        grp = grp[grp['Count'] >= 30]
        segments.append(grp[['Segment', 'ChurnRate', 'Count']])

    result = pd.concat(segments, ignore_index=True)
    result['ChurnRate'] = result['ChurnRate'] * 100
    result = result.sort_values('ChurnRate', ascending=False).drop_duplicates('Segment')
    return result.head(15).reset_index(drop=True)


def get_retention_strategies() -> list:
    """
    Evidence-based retention strategies derived from data patterns.
    Returns list of strategy dicts.
    """
    return [
        {
            "segment": "Month-to-Month + New Customers (0-12m)",
            "churn_driver": "No lock-in, low switching cost, high early dissatisfaction",
            "strategy": "Offer 3-month free upgrade or loyalty discount after 6 months to incentivize conversion to annual plan.",
            "kpi": "Contract upgrade rate, Month-6 NPS",
            "impact": "HIGH",
        },
        {
            "segment": "Fiber Optic + Electronic Check Payers",
            "churn_driver": "High monthly cost ($65-90) with payment friction and billing dissatisfaction",
            "strategy": "Auto-pay enrollment campaign with 5% discount. Offer fiber speed upgrade trial at current price.",
            "kpi": "Auto-pay conversion, ARPU retention",
            "impact": "HIGH",
        },
        {
            "segment": "No Tech Support / No Online Security",
            "churn_driver": "Perceived low value of service; feeling exposed to technical issues",
            "strategy": "Bundle tech support + security for $5/mo (vs $10 separately). Proactive outreach during incidents.",
            "kpi": "Add-on attach rate, support ticket resolution time",
            "impact": "MEDIUM",
        },
        {
            "segment": "Senior Citizens on Month-to-Month",
            "churn_driver": "Fixed income sensitivity to price increases, less digital adoption",
            "strategy": "Senior loyalty programme with fixed-price guarantee for 2 years and dedicated phone support.",
            "kpi": "Senior cohort retention, CSAT",
            "impact": "MEDIUM",
        },
        {
            "segment": "High Tenure (>36m) at Risk",
            "churn_driver": "Long-term customers feel undervalued vs new customer promotions",
            "strategy": "VIP loyalty perks: free equipment upgrades, priority support, anniversary discounts.",
            "kpi": "Loyalty programme enrolment, NPS among 3yr+ cohort",
            "impact": "MEDIUM",
        },
    ]


def print_insights(df: pd.DataFrame):
    """Print formatted business insights to console."""
    print("\n" + "═" * 65)
    print("  📊 BUSINESS INSIGHTS — CHURN ANALYSIS")
    print("═" * 65)

    # Overall
    churn_rate = df['Churn'].mean() * 100
    print(f"\n  Overall Churn Rate: {churn_rate:.1f}%")
    print(f"  Total Customers:    {len(df):,}")
    print(f"  Churned:            {df['Churn'].sum():,}")
    print(f"  Retained:           {(df['Churn'] == 0).sum():,}")

    # Segments
    print("\n  🔴 TOP CHURN RISK SEGMENTS:")
    segments = get_churn_risk_segments(df)
    for i, row in segments.head(5).iterrows():
        print(f"  {i+1}. {row['Segment']:<45} → {row['ChurnRate']:.1f}%  (n={row['Count']:,})")

    # Strategies
    print("\n  💡 RETENTION STRATEGIES:")
    for s in get_retention_strategies():
        print(f"\n  [{s['impact']}] {s['segment']}")
        print(f"       Driver  : {s['churn_driver']}")
        print(f"       Action  : {s['strategy']}")
        print(f"       KPIs    : {s['kpi']}")

    print("\n" + "═" * 65)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw_data, clean_data

    df = load_raw_data()
    df = clean_data(df)
    print_insights(df)
