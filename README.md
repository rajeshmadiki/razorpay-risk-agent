# Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager

## Defensive Fraud-Risk Decision Engine

An end-to-end defensive fraud-risk decision engine for Razorpay AI Buildathon 2026 Track 02. It combines a Random Forest fraud classifier, explicit risk thresholds, a safety gate, explainable risk signals, FastAPI APIs, and a tamper-evident decision audit trail using synthetic transaction data.

[![Streamlit App]](https://razorpay-risk-agent-p5m5zgofer38zdekzzjtgn.streamlit.app/)

> **🌐 Live Web App**:[https://razorpay-risk-agent-p5m5zgofer38zdekzzjtgn.streamlit.app/] (https://razorpay-risk-agent-p5m5zgofer38zdekzzjtgn.streamlit.app/)
> **💻 GitHub Repository**: [https://github.com/rajeshmadiki/razorpay-risk-agent](https://github.com/rajeshmadiki/razorpay-risk-agent)

---

## ⚡ Defense-Only System Notice

This platform is strictly a **defensive fraud-risk decision engine** built for evaluation, benchmarking, and demonstration purposes. It does **not** execute unauthorized payment actions, freeze real Razorpay merchant transactions, execute live customer verification challenges, or claim observed Razorpay production financial savings. All transaction data is synthetically generated for risk modeling.

---

## 📌 Problem Statement

Payment gateways and digital merchants face two competing fraud risk exposure vectors:

1. **False Negatives (Uncaught Fraud)**: Fraudulent transaction attempts that bypass risk controls resulting in direct financial theft, chargeback penalties, dispute fees, and merchant loss.
2. **False Positives (Legitimate Flagged Orders)**: Legitimate customer purchases incorrectly flagged or blocked, creating customer friction, cart abandonment, lost brand loyalty, and unnecessary manual review overhead.

To balance security and checkout experience, this project focuses on:
- **Risk Estimation**: Predicting continuous transaction fraud probability.
- **Explainability**: Identifying top feature risk drivers for every decision.
- **Bounded Operational Decisions**: Policy-driven threshold assignment (`CLEAR`, `ESCALATE`, `HOLD`).
- **False-Positive Cost Analysis**: Synthesizing net defense impact under explicit unit assumptions.
- **Held-Out Evaluation**: Strict 25% stratified test split evaluation.
- **Auditability**: Cryptographically verifiable decision history.

---

## 💡 Solution Overview

The system processes incoming transaction signals through an integrated risk decision pipeline:

```text
Transaction Attempt
        │
        ▼
Feature Engineering (9 Domain Signals)
        │
        ▼
Random Forest Fraud Classifier (fraud-rf-v1)
        │
        ▼
Fraud Probability (P ∈ [0, 1])
        │
        ▼
Risk Threshold Policy (CLEAR < 0.40 <= ESCALATE < 0.75 <= HOLD)
        │
        ▼
Dynamic Safety Gate (Hold Rate Circuit Breaker > 25%)
        │
        ▼
Operational Action Assignment (CLEAR / ESCALATE / HOLD)
        │
        ▼
Explainable Risk Signals & Risk-Driver Attribution
        │
        ▼
Tamper-Evident SHA-256 Audit Chain Logging
```

*Note: The machine learning model produces raw continuous fraud probability signals, while explicit policy controls and safety gates assign operational decisions.*

---

## 🚀 Key Features

- **Fraud-Risk Scoring**: Class-balanced Random Forest classifier predicting continuous fraud probabilities.
- **Explicit Decision Policy**: Strict probability cutoffs mapping scores to bounded operational decisions (`CLEAR`, `ESCALATE`, `HOLD`).
- **Dynamic Safety Gate**: Rate-limiting circuit breaker preventing catastrophic automated blockages by downgrading `HOLD` to `ESCALATE` if running hold rate exceeds 25% (after warm-up).
- **Explainable Risk Signals**: Real-time identification of top risk-driver features per transaction (e.g., `amount_deviation_ratio`).
- **Tamper-Evident SHA-256 Audit Chain**: Sequential SHA-256 cryptographic hash linking across all evaluation logs for tamper detection.
- **FastAPI Backend Service**: Standard REST API architecture exposing system health, scoring, batch processing, audit retrieval, verification, and model metrics.
- **Streamlit Operations Dashboard**: Visual terminal supporting 5 dedicated operational workspaces:
  1. `01 RISK CONSOLE` — Real-time transaction scoring & decision explanation
  2. `02 BATCH ANALYSIS` — Population-level dataset processing & safety gate monitoring
  3. `03 AUDIT LEDGER` — Compliance log inspection & SHA-256 chain verification
  4. `04 SYSTEM ARCHITECTURE` — Complete system flow & component documentation
  5. `05 MODEL EVIDENCE` — Held-out evaluation metrics, confusion matrix & cost analysis

---

## 📐 System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │              Streamlit Frontend (app.py)               │
               │         5 Workspaces Risk Operations Terminal          │
               └──────────────────────────┬─────────────────────────────┘
                                          │  REST API Calls
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │         FastAPI Backend Service (backend/)             │
               │    Swagger OpenAPI (/docs) & REST API v1 Endpoints     │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
       ┌─────────────────────┐ ┌────────────────────┐ ┌─────────────────────┐
       │   FraudAgent Engine │ │   RandomForest ML  │ │ Synthetic Generator │
       │  (src/agent.py)     │ │ (model:fraud-rf-v1)│ │(src/generate_data)  │
       └──────────┬──────────┘ └────────────────────┘ └─────────────────────┘
                  │
                  ▼
       ┌────────────────────────────────────────────────────────┐
       │   Cryptographic SHA-256 Tamper-Evident Audit Ledger    │
       │               (outputs/audit_trail.csv)                │
       └────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Model

The fraud detector is built using a **class-balanced RandomForest** classifier (`fraud-rf-v1` in `src/model.py`).

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)
```

### Feature Pipeline (9 Domain Features)
1. `amount`: Raw transaction currency value.
2. `merchant_avg_amount`: Historical baseline transaction average for target merchant.
3. `amount_deviation_ratio`: Calculated ratio of transaction amount vs merchant average.
4. `hour_of_day`: Hour of transaction initiation (0–23).
5. `is_night`: Binary flag indicating off-hours night transaction (22:00–05:00).
6. `velocity_last_hour`: Transaction frequency count from same customer in prior hour.
7. `location_mismatch`: Binary flag indicating IP vs billing location discrepancy.
8. `device_change`: Binary flag indicating unrecognized hardware device signature.
9. `customer_tenure_days`: Customer account age in days.

---
The model was evaluated on a held-out test set comprising **150 transactions** (25% stratified split from 600 synthetic population records):

| Evaluation Metric | Score | Percentage | Operational Description |
| :--- | :--- | :--- | :--- |
| **Accuracy** | `0.8467` | **84.67%** | Overall test classification accuracy (127/150 correct) |
| **Precision** | `0.5000` | **50.00%** | Ratio of true positive fraud to total predicted fraud (11/22) |
| **Recall** | `0.4783` | **47.83%** | Ratio of true positive fraud detected out of actual fraud (11/23) |
| **F1 Score** | `0.4889` | **48.89%** | Harmonic mean of Precision and Recall |

---


Evaluated on held-out test set ($N = 150$):

| Actual Class \ Predicted Class | Predicted Legitimate | Predicted Fraud | Total Actual |
| :--- | :--- | :--- | :--- |
| **Actual Legitimate** | **TN = 116** (True Legit) | **FP = 11** (Legit Flagged) | 127 |
| **Actual Fraud** | **FN = 12** (Uncaught Fraud) | **TP = 11** (Caught Fraud) | 23 |

### Accuracy Calculation

$$
\text{Accuracy} = \frac{TN + TP}{N}
= \frac{116 + 11}{150}
= \frac{127}{150}
= 0.8467
= 84.67\%
$$

## ⚖️ False-Positive Cost Analysis

To measure operational trade-offs, false-positive friction costs are evaluated against prevented fraud loss:

$$
\text{Illustrative Net Defense Impact}
=
\text{Fraud Loss Prevented}
-
\text{False Positive Cost}
$$

| Cost Analysis Parameter | Metric Value | Operational Assumption / Calculation |
| :--- | :--- | :--- |
| **Held-out Test Transactions** | `150` | 25% Stratified split baseline |
| **True Positives (TP)** | `11` | True fraud correctly caught |
| **False Positives (FP)** | `11` | Legitimate transactions flagged for review |
| **False Negatives (FN)** | `12` | Uncaught fraud resulting in direct loss |
| **Assumed Average Order Value** | `$100.00` | Standard order value assumption |
| **Assumed Chargeback & Fee** | `$15.00` | Penalty fee per uncaught fraud incident |
| **Assumed FP Friction Penalty** | `$5.00` | Unit friction penalty per escalated legit order |
| **Fraud Loss Prevented** | **`$1,265.00`** | `11 TP × ($100 + $15)` saved |
| **False Positive Cost** | **`$55.00`** | `11 FP × $5.00` friction penalty |
| **Illustrative Net Defense Impact** | **`+$1,210.00`** | Illustrative net defense impact under the stated assumptions |
> ⚡ *Illustrative evaluation assumption — not observed Razorpay production savings.*

---

```markdown
## 📜 Auditability & SHA-256 Hash Chain Verification

Every risk decision evaluated by `FraudAgent` exports a cryptographically linked record to `outputs/audit_trail.csv`:

```python
payload = f"{prev_hash}|{timestamp}|{txn_id}|{fraud_prob:.4f}|{orig_decision}|{final_decision}|{override_reason}|{top_feature}"
record_hash = sha256(payload)
## 📡 FastAPI Backend Service Specifications

| Endpoint | Method | Description | Response Schema |
| :--- | :--- | :--- | :--- |
| `/api/v1/health` | `GET` | System health status, version, and model metadata | `HealthResponse` |
| `/api/v1/risk/score` | `POST` | Evaluate single transaction risk score & decision | `RiskScoreResponse` |
| `/api/v1/risk/batch` | `POST` | Execute population batch evaluation across dataset | `BatchResponse` |
| `/api/v1/audit` | `GET` | Query compliance audit records from log | `List[Dict]` |
| `/api/v1/audit/verify` | `GET` | Verify cryptographic SHA-256 audit chain integrity | `AuditVerifyResponse` |
| `/api/v1/evaluation` | `GET` | Fetch held-out model evaluation metrics & importances | `EvaluationResponse` |
| `/api/v1/evaluation/cost` | `GET` | Fetch false-positive cost analysis & net impact | `CostAnalysisResponse` |
| `/docs` | `GET` | Interactive Swagger OpenAPI UI documentation | `HTML` |
| `/openapi.json` | `GET` | Machine-readable OpenAPI spec | `JSON` |
## 💻 Streamlit Operations Dashboard

The Streamlit interface (`app.py`) provides 5 dedicated workspaces:

1. **`01 RISK CONSOLE`**: Single-transaction risk evaluator with feature sliders, quick demo scenarios (Clear, Escalate, Hold), continuous probability meter, decision banner, and decision explanation breakdown.
2. **`02 BATCH ANALYSIS`**: Dataset population risk batch executor, safety circuit breaker status indicator, and intervention distribution breakdown.
3. **`03 AUDIT LEDGER`**: Interactive compliance log viewer with record filtering and 1-click SHA-256 cryptographic chain verification.
4. **`04 SYSTEM ARCHITECTURE`**: Interactive pipeline visualization, component specs, decision policy tables, and evaluator Q&A matrix.
5. **`05 MODEL EVIDENCE`**: Laboratory workspace displaying held-out test set metrics, confusion matrix breakdown, cost analysis, and automated test suite evidence.

---
## 🎭 Demonstration Scenarios

Deterministic demo scenarios provided in the console for evaluation:

- **Scenario 1 — Standard Transaction (`CLEAR`)**: Low amount (`$350.00`), normal velocity, matching location, matching device → Low probability (`P < 0.40`), auto-approved.
- **Scenario 2 — Elevated-Risk Transaction (`ESCALATE`)**: Moderate amount (`$450.00`), night window, minor deviation → Moderate probability (`0.40 ≤ P < 0.75`), routed for additional verification / review.
- **Scenario 3 — High-Risk Transaction (`HOLD`)**: High amount (`$1,280.00`), 6.4x merchant average deviation, night purchase, location mismatch, device change → High probability (`P ≥ 0.75`), assigned a high-risk HOLD decision by the risk policy.
*Note: Demonstration scenarios utilize synthetic input data for verification.*

---

## 🛡️ Safety Control Gate & Defensive Scope

### Safety Circuit Breaker Policy
To guard against automated blockages during false-positive spikes, `FraudAgent` tracks running `HOLD` rate. If the hold rate exceeds **25%** across processed transactions (after a 10-transaction warm-up window), subsequent high-risk items are downgraded from `HOLD` to `ESCALATE` for additional verification / review.

### Scope Disclosures
- **Defensive Focus**: Strictly limited to risk scoring, explainable decisioning, false-positive analysis, and auditability.
- **Non-Invasive**: Does not execute real customer verification challenges, freeze real Razorpay merchant settlements, or perform offensive operations.
- **Operational Meaning**:
  - `ESCALATE`: Assigned as an additional verification / review recommendation.
  - `HOLD`: Assigned as a high-risk HOLD decision produced by the defensive decision engine.

---

## 🧪 Test Coverage & Automated Evidence

The repository contains 13 automated unit and API integration tests:

- **`tests/test_agent.py`** (`5/5 passed`): Validates feature parsing, decision thresholds, safety gate circuit breaker, SHA-256 hash chain generation, and tamper detection.
- **`tests/test_api.py`** (`8/8 passed`): Validates FastAPI `TestClient` REST endpoints (`/health`, `/risk/score`, `/risk/batch`, `/audit`, `/audit/verify`, `/evaluation`, `/evaluation/cost`).

### Automated Test Verification
```text
13 automated unit and API tests
13 passed · 0 failed
Status: OK (outputs/test_summary.json)
```

---

## 📂 Repository Structure

```text
razorpay-risk-agent/
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application & REST endpoint routes
│   ├── schemas.py               # Pydantic request/response schemas & disclosures
│   └── services.py              # RiskEngineService bridging ML & agent core
├── data/
│   └── transactions.csv         # Synthetic transaction dataset (~600 rows)
├── outputs/
│   ├── audit_trail.csv          # Cryptographic SHA-256 decision audit records
│   ├── cost_analysis.json       # False-positive cost analysis & net impact
│   ├── confusion_matrix.json    # Held-out confusion matrix counts
│   ├── evaluation.json          # Held-out model evaluation metrics
│   ├── test_summary.json        # Automated test execution evidence
│   └── threshold_analysis.csv   # Threshold sensitivity analysis matrix
├── src/
│   ├── __init__.py
│   ├── agent.py                 # FraudAgent core, policy engine, safety gate & SHA-256 verifier
│   ├── generate_data.py         # Synthetic transaction generator
│   └── model.py                 # RandomForest model (fraud-rf-v1) training & evaluation
├── tests/
│   ├── __init__.py
│   ├── test_agent.py            # Unit tests for FraudAgent & SHA-256 audit chain
│   └── test_api.py              # Integration tests for FastAPI endpoints
├── app.py                       # Streamlit risk operations dashboard (5 workspaces)
├── README.md                    # Submission documentation & system specification
├── run_tests.py                 # Automated test suite runner & artifact exporter
└── requirements.txt             # Dependency specification
```

---

## 🛠️ Reproducibility & Local Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/rajeshmadiki/razorpay-risk-agent.git
cd razorpay-risk-agent

# Create & activate virtual environment (optional)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model & Export Artifacts

```bash
python src/model.py
```

### 3. Run Automated Test Suite

```bash
python run_tests.py
# Or via standard unittest:
python -m unittest discover -s tests -v
```

### 4. Launch Services

```bash
# Terminal 1: Launch FastAPI Backend Service
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch Streamlit Dashboard
streamlit run app.py
```

---

## ⚠️ System Limitations

- **Synthetic Data**: Dataset (`data/transactions.csv`, 600 records) is synthetically generated for risk modeling demonstration purposes.
- **Held-Out Recall**: Recall on the held-out test set is `47.83%` (11/23 actual fraud detected), reflecting standard un-tuned Random Forest baseline performance on imbalanced risk splits.
- **Illustrative Assumptions**: Unit cost parameters ($100 order value, $15 chargeback fee, $5 friction penalty) are evaluation assumptions and do not represent observed Razorpay production economics.
- **No Production Integration**: Platform operates strictly in demonstration mode without live Razorpay payment processing connection.

---

## 📈 Interpretation of Evaluation Metrics

- **Precision (50.00%)**: Out of 22 total transactions flagged as fraud by the model, 11 were actual fraud.
- **Recall (47.83%)**: Out of 23 actual fraudulent transactions in the held-out test set, 11 were correctly identified by the model.
- **F1 Score (48.89%)**: Balanced metric reflecting harmonic mean of precision and recall.
- **Accuracy (84.67%)**: 127 of 150 held-out test transactions were classified correctly.

---

## 📋 Razorpay Buildathon Track 02 Alignment

| Evaluator Verification Checklist | Implementation & Technical Evidence |
| :--- | :--- |
| **1. Target Loss Class** | Transaction Fraud Risk Loss. |
| **2. Fraud Detection Engine** | Working class-balanced Random Forest classifier (`fraud-rf-v1`). |
| **3. Measured Performance** | Held-out Precision (`50.00%`), Recall (`47.83%`), F1 (`48.89%`), Accuracy (`84.67%`). |
| **4. Held-Out Evaluation** | 25% Stratified test split (150 transactions out of 600). |
| **5. False-Positive Analysis** | 11 FPs ($55 assumed friction cost) vs 11 TPs ($1,265 illustrative prevented-loss value) → +$1,210 illustrative net defense impact. |
| **6. Bounded Decisions** | Operational thresholds (`CLEAR` < 0.40 <= `ESCALATE` < 0.75 <= `HOLD`). |
| **7. Explainability** | Real-time identification of the strongest risk-driving features per transaction.|
| **8. Auditability** | Sequential SHA-256 hash chain logging across 600 verified records with 0 tamper events. |
| **9. REST API** | FastAPI backend service with Swagger docs (`/docs`). |
| **10. UI Terminal** | Deployed Streamlit dashboard with 5 operational workspaces. |
| **11. Safety & Scope** | Defense-only implementation with dynamic rate-limiting safety gate. |

---

## 🎯 Final Verification Summary

```text
Held-out Test Transactions : 150
Precision                  : 0.5000 (50.00%)
Recall                     : 0.4783 (47.83%)
F1 Score                   : 0.4889 (48.89%)
Accuracy                   : 0.8467 (84.67%)
Confusion Matrix           : TN=116 | FP=11 | FN=12 | TP=11
Automated Tests            : 13/13 Passed (0 Failed)
Audit Ledger Verification  : 600/600 SHA-256 Records Verified (0 Tamper Events)
Illustrative Net Impact    : +$1,210.00
```

---

## 🔗 Submission Links

- **Live Application**: [https://razorpay-risk-agent-p5m5zgofer38zdekzzjtgn.streamlit.app/](https://razorpay-risk-agent-p5m5zgofer38zdekzzjtgn.streamlit.app/)
- **GitHub Repository**: [https://github.com/rajeshmadiki/razorpay-risk-agent](https://github.com/rajeshmadiki/razorpay-risk-agent)
- **Razorpay Buildathon**: [https://razorpay.com/buildathon/](https://razorpay.com/buildathon/)

---

> *"Detect the risk, explain the signal, bound the decision, measure the trade-off, and preserve the evidence."*

**Built for Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager.**
