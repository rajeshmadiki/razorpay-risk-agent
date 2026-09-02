import os
import numpy as np
import pandas as pd

def generate_synthetic_transactions(n_samples: int = 600, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic transactions with interpretable fraud indicators and realistic noise.
    """
    np.random.seed(random_seed)

    transaction_ids = [f"TXN_{1000 + i}" for i in range(n_samples)]
    
    # Base features for customers/merchants
    customer_tenure_days = np.random.exponential(scale=300, size=n_samples).astype(int) + 1
    customer_tenure_days = np.clip(customer_tenure_days, 1, 1500)
    
    merchant_avg_amount = np.round(np.random.gamma(shape=3.0, scale=30.0, size=n_samples) + 10.0, 2)
    
    # Generate latent risk propensity (0.0 to 1.0)
    latent_risk = np.random.beta(a=1.5, b=5.0, size=n_samples)

    # Features linked to latent risk with realistic distributions
    # 1. Amount & Deviation Ratio
    amount_deviation_ratio = np.where(
        np.random.rand(n_samples) < latent_risk * 0.8,
        np.random.uniform(2.5, 9.0, size=n_samples),   # High deviation for high risk
        np.random.lognormal(mean=0.0, sigma=0.4, size=n_samples) # Normal deviation (~1.0)
    )
    amount_deviation_ratio = np.round(np.clip(amount_deviation_ratio, 0.1, 15.0), 2)
    amount = np.round(merchant_avg_amount * amount_deviation_ratio, 2)

    # 2. Time features (hour of day & is_night)
    # Higher risk tends towards night hours (22:00 - 05:00)
    night_prob = np.clip(0.15 + latent_risk * 0.6, 0.0, 0.9)
    is_night = (np.random.rand(n_samples) < night_prob).astype(int)
    
    hour_of_day = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if is_night[i] == 1:
            night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
            hour_of_day[i] = np.random.choice(night_hours)
        else:
            day_hours = list(range(6, 22))
            hour_of_day[i] = np.random.choice(day_hours)

    # 3. Velocity in last hour
    velocity_lambda = 1.2 + latent_risk * 4.5
    velocity_last_hour = np.random.poisson(lam=velocity_lambda)
    velocity_last_hour = np.clip(velocity_last_hour, 1, 15)

    # 4. Security mismatch flags
    loc_mismatch_prob = np.clip(0.05 + latent_risk * 0.65, 0.0, 0.85)
    location_mismatch = (np.random.rand(n_samples) < loc_mismatch_prob).astype(int)

    dev_change_prob = np.clip(0.08 + latent_risk * 0.60, 0.0, 0.80)
    device_change = (np.random.rand(n_samples) < dev_change_prob).astype(int)

    # Compute continuous fraud score based on interpretable rules
    risk_score = (
        0.30 * (amount_deviation_ratio > 3.0) +
        0.20 * is_night +
        0.20 * (velocity_last_hour >= 4) +
        0.25 * location_mismatch +
        0.20 * device_change +
        0.15 * (customer_tenure_days < 30)
    )

    # Convert score to target label (~15-20% base fraud rate)
    is_fraud = (risk_score >= 0.55).astype(int)

    # Inject ~6% random label noise to create realistic overlap & non-trivial boundary
    noise_mask = np.random.rand(n_samples) < 0.06
    is_fraud[noise_mask] = 1 - is_fraud[noise_mask]

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "amount": amount,
        "merchant_avg_amount": merchant_avg_amount,
        "amount_deviation_ratio": amount_deviation_ratio,
        "hour_of_day": hour_of_day,
        "is_night": is_night,
        "velocity_last_hour": velocity_last_hour,
        "location_mismatch": location_mismatch,
        "device_change": device_change,
        "customer_tenure_days": customer_tenure_days,
        "is_fraud": is_fraud
    })

    return df

def save_data(output_path: str = "data/transactions.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_transactions()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions and saved to {output_path}")
    print(f"Fraud count: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
    return df

if __name__ == "__main__":
    save_data()
