import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.schemas import (
    HealthResponse,
    TransactionInput,
    RiskScoreResponse,
    BatchResponse,
    EvaluationResponse,
    AuditVerifyResponse
)
from backend.services import RiskEngineService

app = FastAPI(
    title="Razorpay Risk Agent API",
    description="Automated Operational Risk Decisioning & Fraud Verification Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for Streamlit frontend and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["System"])
def root():
    return {
        "message": "Razorpay Risk Agent API Service Online",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Health check endpoint returning real service status and model metadata.
    """
    return HealthResponse(
        status="ONLINE",
        service="Razorpay Risk Agent API",
        version="1.0.0",
        model_name="RandomForestClassifier",
        mode="DEFENSE-ONLY",
        dataset_size=600
    )

@app.post("/api/v1/risk/score", response_model=RiskScoreResponse, tags=["Risk Decisioning"])
def score_transaction(payload: TransactionInput):
    """
    Evaluate real-time risk score and assign operational decision (CLEAR, ESCALATE, HOLD).
    """
    try:
        data_dict = payload.model_dump()
        result = RiskEngineService.evaluate_single(data_dict)
        return RiskScoreResponse(**result)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk Engine Scoring Error: {str(err)}"
        )

@app.post("/api/v1/risk/batch", response_model=BatchResponse, tags=["Batch Operations"])
def process_batch():
    """
    Execute population-level batch risk evaluation across data/transactions.csv.
    """
    try:
        result = RiskEngineService.execute_batch()
        return BatchResponse(**result)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch Execution Error: {str(err)}"
        )

@app.get("/api/v1/audit", response_model=List[Dict[str, Any]], tags=["Audit Evidence"])
def get_audit_trail(
    decision: Optional[str] = Query(None, description="Filter by decision: CLEAR, ESCALATE, HOLD"),
    txn_id: Optional[str] = Query(None, description="Search by Transaction ID substring")
):
    """
    Expose traceable compliance audit records from outputs/audit_trail.csv.
    """
    try:
        return RiskEngineService.get_audit_trail(decision_filter=decision, txn_search=txn_id)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audit Log Retrieval Error: {str(err)}"
        )

@app.get("/api/v1/audit/verify", response_model=AuditVerifyResponse, tags=["Audit Evidence"])
def verify_audit_trail():
    """
    Cryptographically verify the SHA-256 integrity of audit records log for tamper detection.
    """
    try:
        result = RiskEngineService.verify_audit_trail()
        return AuditVerifyResponse(**result)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audit Verification Error: {str(err)}"
        )

@app.get("/api/v1/evaluation", response_model=EvaluationResponse, tags=["Model Intelligence"])
def get_evaluation():
    """
    Expose empirical held-out test set model metrics and feature importance weights.
    """
    try:
        result = RiskEngineService.get_evaluation_metrics()
        return EvaluationResponse(**result)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation Metrics Retrieval Error: {str(err)}"
        )

