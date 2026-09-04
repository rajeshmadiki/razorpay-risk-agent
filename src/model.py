import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

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
MODEL_VERSION = "fraud-rf-v1"

def load_data(filepath: str = "data/transactions.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        from src.generate_data import save_data
        print(f"File {filepath} not found. Generating dataset...")
        return save_data(filepath)
    return pd.read_csv(filepath)

def evaluate_thresholds(y_true, y_proba) -> pd.DataFrame:
    """
    Generate threshold sensitivity analysis across candidate cutoff thresholds.
    """
    thresholds = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    records = []

    for th in thresholds:
        preds = (y_proba >= th).astype(int)
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f = float(f1_score(y_true, preds, zero_division=0))
        acc = float(accuracy_score(y_true, preds))
        intervention_rate = float(preds.mean())

        records.append({
            "threshold": th,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f, 4),
            "accuracy": round(acc, 4),
            "intervention_rate": round(intervention_rate, 4),
            "action_policy": "HOLD" if th >= 0.75 else ("ESCALATE" if th >= 0.40 else "CLEAR")
        })

    return pd.DataFrame(records)

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

    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(val) for val in cm.ravel()]

    print("--- Model Evaluation on Held-out Test Set (25%) ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}\n")

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
        "f1": f1,
        "accuracy": acc,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        },
        "test_size": len(y_test),
        "fraud_count": int(y_test.sum()),
        "model_version": MODEL_VERSION,
        "random_seed": random_state
    }

    return clf, metrics, feature_importance_map, (X_train, X_test, y_train, y_test)

def export_cost_analysis(metrics: Dict[str, Any], output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Calculate and export false-positive cost economics and false-negative exposure.
    """
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    test_size = metrics["test_size"]

    fp_rate = round(fp / (tn + fp) if (tn + fp) > 0 else 0.0, 4)
    fn_rate = round(fn / (tp + fn) if (tp + fn) > 0 else 0.0, 4)

    illustrative_cost_per_fp = 5.0
    illustrative_total_fp_cost = round(fp * illustrative_cost_per_fp, 2)
    illustrative_avg_txn_amount = 100.0
    illustrative_chargeback_fee = 15.0
    unit_fraud_loss = illustrative_avg_txn_amount + illustrative_chargeback_fee

    fn_exposure = round(fn * unit_fraud_loss, 2)
    fraud_prevented = round(tp * unit_fraud_loss, 2)
    net_roi = round(fraud_prevented - illustrative_total_fp_cost, 2)

    cost_data = {
        "model_version": metrics.get("model_version", MODEL_VERSION),
        "test_size": test_size,
        "true_positive_count": tp,
        "true_negative_count": tn,
        "false_positive_count": fp,
        "false_positive_rate": fp_rate,
        "illustrative_cost_per_fp": illustrative_cost_per_fp,
        "illustrative_total_fp_cost": illustrative_total_fp_cost,
        "false_negative_count": fn,
        "false_negative_rate": fn_rate,
        "illustrative_avg_txn_amount": illustrative_avg_txn_amount,
        "illustrative_chargeback_fee": illustrative_chargeback_fee,
        "false_negative_value_exposure": fn_exposure,
        "illustrative_fraud_loss_prevented": fraud_prevented,
        "illustrative_net_defense_roi": net_roi,
        "disclaimer": "Illustrative evaluation assumption — not Razorpay production cost."
    }

    cost_json_path = os.path.join(output_dir, "cost_analysis.json")
    with open(cost_json_path, "w") as f:
        json.dump(cost_data, f, indent=2)

    return cost_data

def export_evaluation_artifacts(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Generate and save evaluation.json, confusion_matrix.json, cost_analysis.json, and threshold_analysis.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    clf, metrics, feature_imp, (X_train, X_test, y_train, y_test) = train_fraud_model(df)

    eval_json_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm_json_path = os.path.join(output_dir, "confusion_matrix.json")
    with open(cm_json_path, "w") as f:
        json.dump(metrics["confusion_matrix"], f, indent=2)

    cost_data = export_cost_analysis(metrics, output_dir)

    # Threshold analysis generated on X_train/y_train validation side
    train_proba = clf.predict_proba(X_train)[:, 1]
    th_df = evaluate_thresholds(y_train, train_proba)
    th_csv_path = os.path.join(output_dir, "threshold_analysis.csv")
    th_df.to_csv(th_csv_path, index=False)

    print(f"Evaluation artifacts exported to {output_dir}/")
    return metrics, th_df, cost_data

if __name__ == "__main__":
    df = load_data()
    clf, metrics, feature_imp, _ = train_fraud_model(df)
    export_evaluation_artifacts(df)


