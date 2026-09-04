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

class EvaluationResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    evaluation_split: str
    feature_importances: Dict[str, float]
