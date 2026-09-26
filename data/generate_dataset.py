"""
Generates a realistic synthetic Telco Customer Churn dataset (~7000 records)
mirroring the Kaggle Telco Customer Churn dataset structure.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043


def generate_telco_churn():
    customer_ids = [f"{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 4))}-{''.join(np.random.choice(list('0123456789'), 4))}" for _ in range(N)]

    gender = np.random.choice(['Male', 'Female'], N)
    senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], N, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], N, p=[0.30, 0.70])

    tenure = np.random.choice(range(0, 73), N)

    phone_service = np.random.choice(['Yes', 'No'], N, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == 'No', 'No phone service',
                              np.random.choice(['Yes', 'No'], N, p=[0.42, 0.58]))

    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], N, p=[0.34, 0.44, 0.22])

    def internet_addon(no_label):
        return np.where(
            internet_service == 'No', no_label,
            np.random.choice(['Yes', 'No'], N, p=[0.43, 0.57])
        )

    online_security = internet_addon('No internet service')
    online_backup = internet_addon('No internet service')
    device_protection = internet_addon('No internet service')
    tech_support = internet_addon('No internet service')
    streaming_tv = internet_addon('No internet service')
    streaming_movies = internet_addon('No internet service')

    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], N, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], N, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        N, p=[0.34, 0.23, 0.22, 0.21]
    )

    monthly_charges = np.round(
        np.where(
            internet_service == 'No', np.random.uniform(18, 30, N),
            np.where(internet_service == 'DSL', np.random.uniform(25, 75, N),
                     np.random.uniform(45, 110, N))
        ), 2
    )

    total_charges = np.where(
        tenure == 0, 0.0,
        np.round(monthly_charges * tenure + np.random.normal(0, 5, N), 2)
    )
    total_charges = np.clip(total_charges, 0, None)

    # Churn probability model (realistic)
    churn_prob = (
        0.05
        + 0.25 * (contract == 'Month-to-month')
        + 0.10 * (internet_service == 'Fiber optic')
        + 0.10 * (payment_method == 'Electronic check')
        + 0.08 * (senior_citizen == 1)
        + 0.08 * (tenure < 6)
        - 0.12 * (tenure > 36)
        - 0.08 * (partner == 'Yes')
        - 0.06 * (online_security == 'Yes')
        - 0.05 * (tech_support == 'Yes')
        - 0.04 * (contract == 'Two year')
        + np.random.normal(0, 0.05, N)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = np.where(np.random.random(N) < churn_prob, 'Yes', 'No')

    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })

    # Inject ~10 missing TotalCharges (new customers)
    missing_idx = np.random.choice(df.index, 10, replace=False)
    df.loc[missing_idx, 'TotalCharges'] = np.nan

    return df


if __name__ == '__main__':
    df = generate_telco_churn()
    df.to_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv', index=False)
    print(f"Dataset saved: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
