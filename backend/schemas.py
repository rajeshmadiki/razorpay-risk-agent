from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., example="ONLINE")
    service: str = Field(..., example="Razorpay Risk Agent API")
    version: str = Field(..., example="1.0.0")
    model_name: str = Field(..., example="RandomForestClassifier")
    mode: str = Field(..., example="DEFENSE-ONLY")
    dataset_size: int = Field(..., example=600)

class TransactionInput(BaseModel):
    transaction_id: str = Field("TXN_9999", example="TXN_1042")
    amount: float = Field(..., gt=0, example=350.0)
    merchant_avg_amount: float = Field(..., gt=0, example=50.0)
    hour_of_day: int = Field(..., ge=0, le=23, example=2)
    velocity_last_hour: int = Field(..., ge=1, le=50, example=5)
    location_mismatch: Any = Field("Yes", example="Yes")
    device_change: Any = Field("Yes", example="Yes")
    customer_tenure_days: int = Field(..., ge=1, example=15)

class RiskScoreResponse(BaseModel):
    transaction_id: str
    amount: float
    merchant_avg_amount: float
    amount_deviation_ratio: float
    hour_of_day: int
    is_night: int
    velocity_last_hour: int
    location_mismatch: int
    device_change: int
    customer_tenure_days: int
    fraud_probability: float
    decision: str
    original_decision: str
    override_reason: str
    risk_level: str
    top_risk_factors: List[str]
    thresholds: Dict[str, str]
    model_identifier: str
    safety_gate_triggered: bool

class BatchResponse(BaseModel):
    total_transactions: int
    clear_count: int
    escalate_count: int
    hold_count: int
    clear_percentage: float
    escalate_percentage: float
    hold_percentage: float
    safety_gate_triggered: bool
    running_hold_rate: float
    limit_hold_rate: float
    audit_chain_valid: bool

class EvaluationResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    evaluation_split: str
    feature_importances: Dict[str, float]

class AuditVerifyResponse(BaseModel):
    is_valid: bool = Field(..., example=True)
    total_records: int = Field(..., example=600)
    tampered_index: Optional[int] = Field(None, example=None)
    message: str = Field(..., example="Audit chain integrity verified. All 600 records cryptographically validated.")

class CostAnalysisResponse(BaseModel):
    model_version: str = Field(..., example="fraud-rf-v1")
    test_size: int = Field(..., example=150)
    true_positive_count: int = Field(..., example=11)
    true_negative_count: int = Field(..., example=116)
    false_positive_count: int = Field(..., example=11)
    false_positive_rate: float = Field(..., example=0.0866)
    illustrative_cost_per_fp: float = Field(..., example=5.0)
    illustrative_total_fp_cost: float = Field(..., example=55.0)
    false_negative_count: int = Field(..., example=12)
    false_negative_rate: float = Field(..., example=0.5217)
    illustrative_avg_txn_amount: float = Field(..., example=100.0)
    illustrative_chargeback_fee: float = Field(..., example=15.0)
    false_negative_value_exposure: float = Field(..., example=1380.0)
    illustrative_fraud_loss_prevented: float = Field(..., example=1265.0)
    illustrative_net_defense_impact: float = Field(..., example=1210.0)
    disclaimer: str = Field(..., example="Illustrative evaluation assumption — not observed Razorpay production savings.")


