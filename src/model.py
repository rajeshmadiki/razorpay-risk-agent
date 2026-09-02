import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

FEATURE_COLS = [
    "amount",
    "merchant_avg_amount",
    "amount_deviation_ratio",
    "hour_of_day",
    "is_night",
    "velocity_last_hour",
    "location_mismatch",
    "device_change",
    "customer_tenure_days"
]

TARGET_COL = "is_fraud"

def load_data(filepath: str = "data/transactions.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        from src.generate_data import save_data
        print(f"File {filepath} not found. Generating dataset...")
        return save_data(filepath)
    return pd.read_csv(filepath)

def train_fraud_model(df: pd.DataFrame, random_state: int = 42):
    """
    Train RandomForestClassifier with stratified train/test split and class weighting.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=random_state
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("--- Model Evaluation on Held-out Test Set (25%) ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}\n")

    # Print Feature Importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("--- Feature Importances ---")
    feature_importance_map = {}
    for i in indices:
        feat_name = FEATURE_COLS[i]
        feat_imp = importances[i]
        feature_importance_map[feat_name] = float(feat_imp)
        print(f"  {feat_name:<25}: {feat_imp:.4f}")
    print()

    metrics = {
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    return clf, metrics, feature_importance_map, (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    df = load_data()
    clf, metrics, feature_imp, _ = train_fraud_model(df)
