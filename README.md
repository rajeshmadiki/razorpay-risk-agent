# 🛡️ Razorpay AI Risk Manager
**Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager**

An end-to-end, submission-ready AI Risk Management & Fraud Auto-Responder platform. Built with a **FastAPI backend microservice**, a **Streamlit visual dashboard**, a class-balanced **RandomForest risk model**, defensive policy enforcement, a **rate-limiting safety gate**, and structured audit trail logging.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://razorpay-risk-agent.streamlit.app/)

> **🌐 Live Web App**: [https://razorpay-risk-agent.streamlit.app/](https://razorpay-risk-agent.streamlit.app/)

---

## 📐 System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │              Streamlit Frontend (app.py)               │
               │   Cinematic Dark/Light Theme Fintech Risk Console       │
               └──────────────────────────┬─────────────────────────────┘
                                          │  REST API Calls
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │               FastAPI Backend (backend/)               │
               │    Swagger OpenAPI (/docs) & REST API v1 Endpoints      │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
       ┌─────────────────────┐ ┌────────────────────┐ ┌─────────────────────┐
       │   FraudAgent Engine │ │   RandomForest ML  │ │ Synthetic Generator │
       │  (src/agent.py)     │ │   (src/model.py)   │ │(src/generate_data)  │
       └──────────┬──────────┘ └────────────────────┘ └─────────────────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │     Audit Trail     │
       │(outputs/audit_trail)│
       └─────────────────────┘
```

### Key Components

- **FastAPI Microservice Backend (`backend/`)**: Standard REST API architecture separating business/risk logic from UI presentation.
- **RandomForest Fraud Engine (`src/model.py`)**: Trained on synthetic financial transaction data (~600 records) with domain-specific risk signals.
- **Defensive Decision Engine (`src/agent.py`)**: Classifies risk into explicit action policies (`CLEAR`, `ESCALATE`, `HOLD`), evaluates feature risk contributions, and writes immutable audit logs.
- **Dynamic Safety Gate**: Automatic rate-limiting circuit breaker that converts excessive `HOLD` decisions to `ESCALATE` if running hold rate exceeds 25%.
- **Streamlit Operations Dashboard (`app.py`)**: Real-time risk analysis console with seamless theme-safe CSS (Light & Dark modes), REST API client, and local engine fallback.

---

## 📂 Project Structure

```text
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI server & route handlers
│   ├── schemas.py               # Pydantic request & response schemas
│   └── services.py              # RiskEngineService bridging ML & agent core
├── data/
│   └── transactions.csv         # Synthetic transaction dataset (~600 rows)
├── outputs/
│   └── audit_trail.csv          # Agent evaluation logs & feature explanations
├── src/
│   ├── __init__.py
│   ├── generate_data.py         # Synthetic transaction dataset generator
│   ├── model.py                 # RandomForest training & evaluation
│   └── agent.py                 # FraudAgent core, decision policy & safety gate
├── tests/
│   ├── __init__.py
│   ├── test_agent.py            # Standard unittest suite for agent policy
│   └── test_api.py              # FastAPI endpoint integration unit tests
├── app.py                       # Streamlit risk operations dashboard
├── README.md                    # Submission documentation
└── requirements.txt             # Dependency specification
```

---

## ⚡ Quick Start & Execution Guide

### 1. Environment Setup

```bash
# Clone the repository and navigate into the folder
cd razorpay

# Create virtual environment (optional)
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend Service

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API Server**: Runs at `http://localhost:8000`
- **Interactive Swagger Docs**: Available at `http://localhost:8000/docs`
- **ReDoc API Documentation**: Available at `http://localhost:8000/redoc`

### 3. Launch Streamlit Frontend Dashboard

In a separate terminal:
```bash
streamlit run app.py
```
- **Dashboard URL**: `http://localhost:8501`

### 4. Run Test Suite

```bash
python -m unittest discover -s tests
```
Runs 10 automated unit & API test cases verifying health endpoints, single scoring, batch evaluation, audit fetching, and safety gate triggers.

---

## 📡 REST API Specifications (`/api/v1`)

| Endpoint | Method | Description | Request Payload / Params |
| :--- | :--- | :--- | :--- |
| `/api/v1/health` | `GET` | System status, model loading, dataset record count | None |
| `/api/v1/risk/score` | `POST` | Evaluate a single transaction & compute action | `TransactionInput` |
| `/api/v1/risk/batch` | `POST` | Evaluate entire batch or uploaded CSV | Optional CSV payload |
| `/api/v1/audit` | `GET` | Fetch recent audit trail logs & filter by decision | `decision` (optional query) |
| `/api/v1/evaluation` | `GET` | Retrieve held-out model evaluation metrics | None |

---

## 🎯 Decision Policy & Safety Gate Rules

| Risk Score ($P$) | Default Action | Safety Gate Action (Hold Rate > 25%) | Operational Description |
| :--- | :--- | :--- | :--- |
| **$P \ge 0.75$** | `HOLD` | Downgraded to `ESCALATE` | High risk — transaction frozen for intervention |
| **$0.40 \le P < 0.75$** | `ESCALATE` | `ESCALATE` | Moderate risk — routed for 2FA / step-up verification |
| **$P < 0.40$** | `CLEAR` | `CLEAR` | Low risk — transaction auto-approved |

> **🛡️ Safety Circuit Breaker**: To prevent catastrophic automated blockages during false-positive spikes or malicious traffic bursts, the `FraudAgent` tracks the running `HOLD` rate. If the hold rate exceeds **25%** across processed transactions (after a 10-transaction warm-up window), all subsequent high-risk transactions are downgraded from `HOLD` to `ESCALATE` for human verification.

---

## 📊 Model Evaluation & Operational Performance

Evaluated on held-out test dataset (150 transactions / 25% stratified split):

### Evaluation Metrics
| Metric | Score | Percentage | Description |
# 🛡️ Razorpay AI Risk Manager
**Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager**

An end-to-end, submission-ready AI Risk Management & Fraud Auto-Responder platform. Built with a **FastAPI backend microservice**, a **Streamlit visual dashboard**, a class-balanced **RandomForest risk model**, defensive policy enforcement, a **rate-limiting safety gate**, and structured audit trail logging.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://razorpay-risk-agent.streamlit.app/)

> **🌐 Live Web App**: [https://razorpay-risk-agent.streamlit.app/](https://razorpay-risk-agent.streamlit.app/)

---

## 📐 System Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │              Streamlit Frontend (app.py)               │
               │   Cinematic Dark/Light Theme Fintech Risk Console       │
               └──────────────────────────┬─────────────────────────────┘
                                          │  REST API Calls
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │               FastAPI Backend (backend/)               │
               │    Swagger OpenAPI (/docs) & REST API v1 Endpoints      │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
       ┌─────────────────────┐ ┌────────────────────┐ ┌─────────────────────┐
       │   FraudAgent Engine │ │   RandomForest ML  │ │ Synthetic Generator │
       │  (src/agent.py)     │ │   (src/model.py)   │ │(src/generate_data)  │
       └──────────┬──────────┘ └────────────────────┘ └─────────────────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │     Audit Trail     │
       │(outputs/audit_trail)│
       └─────────────────────┘
```

### Key Components

- **FastAPI Microservice Backend (`backend/`)**: Standard REST API architecture separating business/risk logic from UI presentation.
- **RandomForest Fraud Engine (`src/model.py`)**: Trained on synthetic financial transaction data (~600 records) with domain-specific risk signals.
- **Defensive Decision Engine (`src/agent.py`)**: Classifies risk into explicit action policies (`CLEAR`, `ESCALATE`, `HOLD`), evaluates feature risk contributions, and writes immutable audit logs.
- **Dynamic Safety Gate**: Automatic rate-limiting circuit breaker that converts excessive `HOLD` decisions to `ESCALATE` if running hold rate exceeds 25%.
- **Streamlit Operations Dashboard (`app.py`)**: Real-time risk analysis console with seamless theme-safe CSS (Light & Dark modes), REST API client, and local engine fallback.

---

## 📂 Project Structure

```text
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI server & route handlers
│   ├── schemas.py               # Pydantic request & response schemas
│   └── services.py              # RiskEngineService bridging ML & agent core
├── data/
│   └── transactions.csv         # Synthetic transaction dataset (~600 rows)
├── outputs/
│   └── audit_trail.csv          # Agent evaluation logs & feature explanations
├── src/
│   ├── __init__.py
│   ├── generate_data.py         # Synthetic transaction dataset generator
│   ├── model.py                 # RandomForest training & evaluation
│   └── agent.py                 # FraudAgent core, decision policy & safety gate
├── tests/
│   ├── __init__.py
│   ├── test_agent.py            # Standard unittest suite for agent policy
│   └── test_api.py              # FastAPI endpoint integration unit tests
├── app.py                       # Streamlit risk operations dashboard
├── README.md                    # Submission documentation
└── requirements.txt             # Dependency specification
```

---

## ⚡ Quick Start & Execution Guide

### 1. Environment Setup

```bash
# Clone the repository and navigate into the folder
cd razorpay

# Create virtual environment (optional)
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend Service

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API Server**: Runs at `http://localhost:8000`
- **Interactive Swagger Docs**: Available at `http://localhost:8000/docs`
- **ReDoc API Documentation**: Available at `http://localhost:8000/redoc`

### 3. Launch Streamlit Frontend Dashboard

In a separate terminal:
```bash
streamlit run app.py
```
- **Dashboard URL**: `http://localhost:8501`

### 4. Run Test Suite

```bash
python -m unittest discover -s tests
```
Runs 10 automated unit & API test cases verifying health endpoints, single scoring, batch evaluation, audit fetching, and safety gate triggers.

---

## 📡 REST API Specifications (`/api/v1`)

| Endpoint | Method | Description | Request Payload / Params |
| :--- | :--- | :--- | :--- |
| `/api/v1/health` | `GET` | System status, model loading, dataset record count | None |
| `/api/v1/risk/score` | `POST` | Evaluate a single transaction & compute action | `TransactionInput` |
| `/api/v1/risk/batch` | `POST` | Evaluate entire batch or uploaded CSV | Optional CSV payload |
| `/api/v1/audit` | `GET` | Fetch recent audit trail logs & filter by decision | `decision` (optional query) |
| `/api/v1/evaluation` | `GET` | Retrieve held-out model evaluation metrics | None |

---

## 🎯 Decision Policy & Safety Gate Rules

| Risk Score ($P$) | Default Action | Safety Gate Action (Hold Rate > 25%) | Operational Description |
| :--- | :--- | :--- | :--- |
| **$P \ge 0.75$** | `HOLD` | Downgraded to `ESCALATE` | High risk — transaction frozen for intervention |
| **$0.40 \le P < 0.75$** | `ESCALATE` | `ESCALATE` | Moderate risk — routed for 2FA / step-up verification |
| **$P < 0.40$** | `CLEAR` | `CLEAR` | Low risk — transaction auto-approved |

> **🛡️ Safety Circuit Breaker**: To prevent catastrophic automated blockages during false-positive spikes or malicious traffic bursts, the `FraudAgent` tracks the running `HOLD` rate. If the hold rate exceeds **25%** across processed transactions (after a 10-transaction warm-up window), all subsequent high-risk transactions are downgraded from `HOLD` to `ESCALATE` for human verification.

---

## 📊 Model Evaluation & Operational Performance

Evaluated on held-out test dataset (150 transactions / 25% stratified split):

### Evaluation Metrics
| Metric | Score | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **Accuracy** | `0.8500` | 85.00% | Overall test classification accuracy |
| **Precision** | `0.5000` | 50.00% | Ratio of true positive fraud to total predicted fraud |
| **Recall** | `0.4783` | 47.83% | Ratio of true positive fraud detected out of total actual fraud |
| **F1 Score** | `0.4889` | 48.89% | Harmonic mean of Precision and Recall |

### Top Risk Driver Feature Importances
1. `amount_deviation_ratio`: **29.06%**
2. `amount`: **26.47%**
3. `merchant_avg_amount`: **10.51%**
4. `customer_tenure_days`: **7.51%**
5. `hour_of_day`: **7.17%**
6. `location_mismatch`: **6.76%**
7. `velocity_last_hour`: **5.30%**
8. `is_night`: **4.71%**
9. `device_change`: **2.51%**

### Full Dataset Batch Distribution (600 Transactions)
- **CLEAR**: 477 (79.5%)
- **ESCALATE**: 62 (10.3%)
- **HOLD**: 61 (10.2%)
- **Safety Gate**: Active (Hold rate 10.2% $\le$ 25% threshold)

---

## ⚖️ False-Positive Cost & Merchant Loss Analysis

| Category | Type | Metric / Value | Operational Description |
| :--- | :--- | :--- | :--- |
| **Held-out Test Transactions** | `MEASURED` | `150` transactions | 25% Stratified split evaluation baseline |
| **True Positives (TP)** | `MEASURED` | `11` transactions | True fraud correctly caught and prevented |
| **False Positives (FP)** | `MEASURED` | `11` transactions | Legitimate transactions flagged for step-up review |
| **False Negatives (FN)** | `MEASURED` | `12` transactions | Uncaught fraud resulting in direct loss |
| **Average Transaction Amount** | `ASSUMED` | `$100.00` | Standard order value assumption |
| **Chargeback & Penalty Fee** | `ASSUMED` | `$15.00` | Fee per uncaught fraud incident |
| **FP Friction Penalty Cost** | `ASSUMED` | `$5.00` | Customer friction impact per escalated legit order |
| **Fraud Loss Prevented** | `SYNTHESIZED` | **`$1,265.00`** | `11 TP × ($100 + $15)` saved |
| **False Positive Cost** | `SYNTHESIZED` | **`$55.00`** | `11 FP × $5.00` friction cost |
| **Net Defense ROI** | `SYNTHESIZED` | **`+$1,210.00 Net`** | Direct net merchant loss reduction |

---

## 📋 Evaluator Checklist & Track 02 Alignment

| Evaluator Verification Question | Implementation & Evidence |
| :--- | :--- |
| **1. What problem is being solved?** | Stopping merchant financial loss from transaction fraud, returns, and chargebacks. |
| **2. What type of loss is targeted?** | Transaction Fraud Risk Loss. |
| **3. Why is fraud the selected loss class?** | Fraud represents direct monetary loss through stolen card usage, unauthorized payments, and high chargeback fees. |
| **4. What exactly does the system detect?** | High-risk, anomalous payment attempts before funds settle. |
| **5. What signals does it evaluate?** | 9 domain features: `amount_deviation_ratio`, `is_night`, `velocity_last_hour`, `location_mismatch`, `device_change`, `customer_tenure_days`, `amount`, `merchant_avg_amount`, `hour_of_day`. |
| **6. How is the risk score generated?** | Class-balanced `RandomForest` classifier predicts continuous raw fraud probability $P \in [0, 1]$. |
| **7. How does the decision policy work?** | Strict thresholding mapping raw probability to operational action rules. |
| **8. What happens for CLEAR?** | $P < 0.40 \rightarrow$ Transaction auto-approved without customer friction. |
| **9. What happens for ESCALATE?** | $0.40 \le P < 0.75 \rightarrow$ Step-up 2FA / OTP verification requested. |
| **10. What happens for HOLD?** | $P \ge 75\% \rightarrow$ High-risk payment frozen for manual intervention. |
| **11. Is there a held-out test set?** | Yes, 25% stratified test split (150 transactions out of 600). |
| **12. What are precision and recall?** | Precision: **`0.5000`** (50.0%), Recall: **`0.4783`** (47.83%), F1: **`0.4889`**, Accuracy: **`0.8500`**. |
| **13. What is the false-positive cost?** | Measured 11 FPs out of 150 test transactions ($55.00 assumed friction cost vs $1,265.00 fraud prevented). |
| **14. What is measured vs assumed?** | `MEASURED` = classification counts, precision, recall, F1, accuracy, hold rate. `ASSUMED` = avg transaction value ($100), chargeback fee ($15), friction penalty ($5). |
| **15. Is there a safety gate?** | Yes, automatic circuit breaker downgrading `HOLD` to `ESCALATE` if running hold rate exceeds 25% (after 10-txn warm-up). |
| **16. Can the decision be audited?** | Yes, every evaluation logs timestamp, transaction ID, score, action, and top risk driver to `outputs/audit_trail.csv`. |
| **17. Is the system defense-only?** | Yes, strictly defensive risk scoring, decisioning, and rate-limiting. Zero offensive or exploit capabilities. |
| **18. Can the evaluator run it locally?** | Yes (`uvicorn backend.main:app --port 8000` & `streamlit run app.py`). |
| **19. Can the evaluator inspect backend/API?** | Yes, Swagger OpenAPI docs live at `http://localhost:8000/docs` and REST endpoints under `/api/v1`. |

---

## 🧪 Testing & Verification

The project includes unit tests for both backend APIs and risk core:

- `tests/test_agent.py`: Validates `FraudAgent` scoring, decision thresholds, feature ranking, safety gate circuit breaker, and audit output.
- `tests/test_api.py`: Validates FastAPI `TestClient` REST endpoints (`/health`, `/risk/score`, `/risk/batch`, `/audit`, `/evaluation`).

Run all tests via standard python test discovery:
```bash
python -m unittest discover -s tests
```

---

## 📜 License & Acknowledgments

Developed for **Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager**.
