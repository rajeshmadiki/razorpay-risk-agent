import os
import sys
import pandas as pd
from typing import Tuple, Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import load_data, train_fraud_model, FEATURE_COLS
from src.agent import FraudAgent, run_agent_batch, verify_audit_chain

class RiskEngineService:
    _instance = None
    _agent: FraudAgent = None
    _dataset: pd.DataFrame = None
    _metrics: Dict[str, float] = None
    _feature_importances: Dict[str, float] = None

    @classmethod
    def get_agent(cls) -> Tuple[FraudAgent, Dict[str, float], Dict[str, float], pd.DataFrame]:
        if cls._agent is None:
            data_path = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
            cls._dataset = load_data(data_path)
            model, metrics, feat_importances, (X_train, X_test, y_train, y_test) = train_fraud_model(cls._dataset)
            cls._metrics = metrics
            cls._feature_importances = feat_importances
            cls._agent = FraudAgent(
                model=model,
                feature_importances=feat_importances,
                feature_means=X_train.mean(),
                feature_stds=X_train.std()
            )
        return cls._agent, cls._metrics, cls._feature_importances, cls._dataset

    @classmethod
    def evaluate_single(cls, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        agent, metrics, feat_importances, df = cls.get_agent()
        
        amount = float(data_dict["amount"])
        merchant_avg = float(data_dict["merchant_avg_amount"])
        dev_ratio = round(amount / merchant_avg if merchant_avg > 0 else 1.0, 2)
        hour = int(data_dict["hour_of_day"])
        is_night = 1 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
        velocity = int(data_dict["velocity_last_hour"])
        
        loc_val = data_dict.get("location_mismatch")
        loc_mismatch = 1 if loc_val in [1, "1", "Yes", True] else 0
        
        dev_val = data_dict.get("device_change")
        dev_change = 1 if dev_val in [1, "1", "Yes", True] else 0
        
        tenure = int(data_dict["customer_tenure_days"])
        txn_id = str(data_dict.get("transaction_id", "TXN_9999"))

        input_row = pd.Series({
            "transaction_id": txn_id,
            "amount": amount,
            "merchant_avg_amount": merchant_avg,
            "amount_deviation_ratio": dev_ratio,
            "hour_of_day": hour,
            "is_night": is_night,
            "velocity_last_hour": velocity,
            "location_mismatch": loc_mismatch,
            "device_change": dev_change,
            "customer_tenure_days": tenure
        })

        prob, decision, top_feature, orig_decision, override_reason = agent.evaluate_transaction(input_row, is_stateful=False)

        risk_level = "HIGH RISK" if decision == "HOLD" else ("MEDIUM RISK" if decision == "ESCALATE" else "LOW RISK")
        
        top_signals = [top_feature]
        if dev_ratio > 3.0 and "amount_deviation_ratio" not in top_signals:
            top_signals.append("amount_deviation_ratio")
        if velocity > 3 and "velocity_last_hour" not in top_signals:
            top_signals.append("velocity_last_hour")
        if loc_mismatch == 1 and "location_mismatch" not in top_signals:
            top_signals.append("location_mismatch")

        return {
            "transaction_id": txn_id,
            "amount": amount,
            "merchant_avg_amount": merchant_avg,
            "amount_deviation_ratio": dev_ratio,
            "hour_of_day": hour,
            "is_night": is_night,
            "velocity_last_hour": velocity,
            "location_mismatch": loc_mismatch,
            "device_change": dev_change,
            "customer_tenure_days": tenure,
            "fraud_probability": round(prob, 4),
            "decision": decision,
            "original_decision": orig_decision,
            "override_reason": override_reason,
            "risk_level": risk_level,
            "top_risk_factors": top_signals,
            "thresholds": {
                "CLEAR": "< 0.40",
                "ESCALATE": "0.40 - 0.749",
                "HOLD": ">= 0.75"
            },
            "model_identifier": "RandomForestClassifier",
            "safety_gate_triggered": agent.gate_triggered
        }

    @classmethod
    def execute_batch(cls) -> Dict[str, Any]:
        data_path = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
        output_path = os.path.join(PROJECT_ROOT, "outputs", "audit_trail.csv")
        batch_agent, audit_df = run_agent_batch(data_path, output_path)

        total_txns = len(audit_df)
        holds = int((audit_df["decision"] == "HOLD").sum())
        escalates = int((audit_df["decision"] == "ESCALATE").sum())
        clears = int((audit_df["decision"] == "CLEAR").sum())

        hold_rate = holds / total_txns if total_txns > 0 else 0.0
        verify_res = verify_audit_chain(audit_df)

        return {
            "total_transactions": total_txns,
            "clear_count": clears,
            "escalate_count": escalates,
            "hold_count": holds,
            "clear_percentage": round((clears / total_txns) * 100, 2),
            "escalate_percentage": round((escalates / total_txns) * 100, 2),
            "hold_percentage": round((holds / total_txns) * 100, 2),
            "safety_gate_triggered": batch_agent.gate_triggered,
            "running_hold_rate": round(hold_rate, 4),
            "limit_hold_rate": 0.25,
            "audit_chain_valid": verify_res.get("is_valid", False)
        }

    @classmethod
    def get_audit_trail(cls, decision_filter: Optional[str] = None, txn_search: Optional[str] = None) -> List[Dict[str, Any]]:
        audit_file = os.path.join(PROJECT_ROOT, "outputs", "audit_trail.csv")
        if not os.path.exists(audit_file):
            return []
        
        audit_df = pd.read_csv(audit_file)
        if decision_filter and decision_filter.upper() != "ALL":
            audit_df = audit_df[audit_df["decision"] == decision_filter.upper()]
        
        if txn_search and txn_search.strip():
            audit_df = audit_df[audit_df["transaction_id"].astype(str).str.contains(txn_search.strip(), case=False)]
            
        return audit_df.to_dict(orient="records")

    @classmethod
    def verify_audit_trail(cls) -> Dict[str, Any]:
        output_path = os.path.join(PROJECT_ROOT, "outputs", "audit_trail.csv")
        return verify_audit_chain(output_path)

    @classmethod
    def get_evaluation_metrics(cls) -> Dict[str, Any]:
        agent, metrics, feat_importances, df = cls.get_agent()
        return {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1_score": round(metrics["f1"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "evaluation_split": "25% Stratified Held-out Test Set",
            "feature_importances": {k: round(v, 4) for k, v in feat_importances.items()}
        }

    @classmethod
    def get_cost_analysis(cls) -> Dict[str, Any]:
        cost_file = os.path.join(PROJECT_ROOT, "outputs", "cost_analysis.json")
        if os.path.exists(cost_file):
            import json
            with open(cost_file, "r") as f:
                return json.load(f)

        agent, metrics, feat_importances, df = cls.get_agent()
        from src.model import export_cost_analysis
        return export_cost_analysis(metrics)


