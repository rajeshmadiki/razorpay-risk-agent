import os
import sys
import datetime
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import load_data, train_fraud_model, FEATURE_COLS

class FraudAgent:
    def __init__(self, model, feature_importances: Dict[str, float], feature_means: pd.Series, feature_stds: pd.Series):
        self.model = model
        self.feature_importances = feature_importances
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        
        # State tracking for safety gate
        self.total_processed = 0
        self.total_holds = 0
        self.gate_triggered = False
        self.gate_triggered_at = None

    def calculate_top_contributing_feature(self, row: pd.Series) -> str:
        """
        Determine the top feature driving risk for a given transaction row
        by combining feature importances with normalized feature values.
        """
        risk_contributions = {}
        for col in FEATURE_COLS:
            val = float(row[col])
            mean = float(self.feature_means[col])
            std = float(self.feature_stds[col]) if self.feature_stds[col] > 0 else 1.0
            importance = self.feature_importances.get(col, 0.1)

            # Direction of risk per feature
            if col == "customer_tenure_days":
                # Lower tenure means higher risk
                z_score = (mean - val) / std
            else:
                # Higher values mean higher risk
                z_score = (val - mean) / std

            # Weight by model feature importance
            risk_contributions[col] = importance * max(z_score, 0.0)

        # Select feature with highest positive risk contribution
        top_feature = max(risk_contributions, key=risk_contributions.get)
        return top_feature

    def evaluate_transaction(self, row: pd.Series, is_stateful: bool = True) -> Tuple[float, str, str, str, str]:
        """
        Score transaction with predict_proba, apply policy and safety gate.
        Returns: (probability, final_decision, top_contributing_feature, original_decision, override_reason)
        """
        features = row[FEATURE_COLS].to_frame().T
        prob = float(self.model.predict_proba(features)[0, 1])

        # Initial raw policy decision
        if prob >= 0.75:
            raw_decision = "HOLD"
        elif prob >= 0.40:
            raw_decision = "ESCALATE"
        else:
            raw_decision = "CLEAR"

        original_decision = raw_decision
        final_decision = raw_decision
        override_reason = "none"

        if is_stateful:
            if raw_decision == "HOLD":
                # Safety gate: track running auto-HOLD rate through the batch.
                # Require minimum warm-up window of 10 processed items before rate evaluation
                # to avoid false activation on initial items (e.g., 1/1 = 100%).
                running_hold_rate = (self.total_holds / self.total_processed) if self.total_processed >= 10 else 0.0

                if self.gate_triggered or (self.total_processed >= 10 and running_hold_rate > 0.25):
                    if not self.gate_triggered:
                        self.gate_triggered = True
                        self.gate_triggered_at = self.total_processed + 1
                    final_decision = "ESCALATE"
                    override_reason = "safety_gate"
                else:
                    self.total_holds += 1

            self.total_processed += 1

        top_feature = self.calculate_top_contributing_feature(row)

        return prob, final_decision, top_feature, original_decision, override_reason

def verify_audit_chain(audit_input: Union[str, pd.DataFrame] = "outputs/audit_trail.csv") -> Dict[str, Any]:
    """
    Cryptographically verify the SHA-256 audit chain log for tamper detection.
    """
    if isinstance(audit_input, str):
        if not os.path.exists(audit_input):
            return {
                "is_valid": False,
                "total_records": 0,
                "tampered_index": None,
                "message": f"Audit trail file not found at {audit_input}"
            }
        audit_df = pd.read_csv(audit_input)
    else:
        audit_df = audit_input

    if audit_df.empty:
        return {
            "is_valid": True,
            "total_records": 0,
            "tampered_index": None,
            "message": "Audit trail is empty."
        }

    if "record_hash" not in audit_df.columns:
        return {
            "is_valid": False,
            "total_records": len(audit_df),
            "tampered_index": 0,
            "message": "Legacy audit trail missing cryptographic SHA-256 hashes."
        }

    prev_hash = "GENESIS"
    for idx, row in audit_df.iterrows():
        timestamp = str(row["timestamp"])
        txn_id = str(row["transaction_id"])
        prob = float(row["fraud_probability"])
        orig_dec = str(row.get("original_decision", row["decision"]))
        final_dec = str(row.get("final_decision", row["decision"]))
        override_reason = str(row.get("override_reason", "none"))
        top_feat = str(row["top_contributing_feature"])
        stored_hash = str(row["record_hash"])

        payload = f"{prev_hash}|{timestamp}|{txn_id}|{prob:.4f}|{orig_dec}|{final_dec}|{override_reason}|{top_feat}"
        calculated_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        if calculated_hash != stored_hash:
            return {
                "is_valid": False,
                "total_records": len(audit_df),
                "tampered_index": int(idx),
                "message": f"Tamper detected at row index {idx} (Txn: {txn_id})"
            }
        prev_hash = calculated_hash

    return {
        "is_valid": True,
        "total_records": len(audit_df),
        "tampered_index": None,
        "message": f"Audit chain integrity verified. All {len(audit_df)} records cryptographically validated."
    }

def run_agent_batch(input_filepath: str = "data/transactions.csv", output_filepath: str = "outputs/audit_trail.csv"):
    df = load_data(input_filepath)
    model, metrics, feat_importances, (X_train, X_test, y_train, y_test) = train_fraud_model(df)

    feature_means = X_train.mean()
    feature_stds = X_train.std()

    agent = FraudAgent(
        model=model,
        feature_importances=feat_importances,
        feature_means=feature_means,
        feature_stds=feature_stds
    )

    audit_records = []
    decision_counts = {"HOLD": 0, "ESCALATE": 0, "CLEAR": 0}
    prev_hash = "GENESIS"

    print("--- Executing Fraud Agent Batch Verification ---")
    for idx, row in df.iterrows():
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        prob, decision, top_feature, orig_decision, override_reason = agent.evaluate_transaction(row, is_stateful=True)
        
        decision_counts[decision] += 1

        payload = f"{prev_hash}|{timestamp}|{row['transaction_id']}|{prob:.4f}|{orig_decision}|{decision}|{override_reason}|{top_feature}"
        record_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        prev_hash = record_hash

        audit_records.append({
            "timestamp": timestamp,
            "transaction_id": row["transaction_id"],
            "fraud_probability": round(prob, 4),
            "original_decision": orig_decision,
            "final_decision": decision,
            "decision": decision,
            "override_reason": override_reason,
            "top_contributing_feature": top_feature,
            "record_hash": record_hash
        })

    audit_df = pd.DataFrame(audit_records)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    audit_df.to_csv(output_filepath, index=False)

    print(f"\nSaved audit log to {output_filepath}")
    print("\n--- Execution Summary ---")
    print(f"Total Transactions Processed: {len(df)}")
    print(f"  HOLD     : {decision_counts['HOLD']} ({decision_counts['HOLD']/len(df)*100:.1f}%)")
    print(f"  ESCALATE : {decision_counts['ESCALATE']} ({decision_counts['ESCALATE']/len(df)*100:.1f}%)")
    print(f"  CLEAR    : {decision_counts['CLEAR']} ({decision_counts['CLEAR']/len(df)*100:.1f}%)")
    print(f"\nSafety Gate Triggered: {agent.gate_triggered}")
    if agent.gate_triggered:
        print(f"  Gate activated at item index: {agent.gate_triggered_at}")

    # Verify chain immediately after export
    verify_res = verify_audit_chain(audit_df)
    print(f"\nCryptographic Verification: {verify_res['message']}")

    return agent, audit_df

if __name__ == "__main__":
    run_agent_batch()

