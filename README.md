# Fraud Verifier & Auto-Responder Agent

An end-to-end Python system for automated fraud decisioning, auto-response policy enforcement, and audit trail logging. Unlike standard classifiers that only output probabilities, this agent enforces operational decision rules, risk explanations, and a defensive rate-limiting safety gate.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://razorpay-risk-agent.streamlit.app/)

> **🌐 Live Web App**: [https://razorpay-risk-agent.streamlit.app/](https://razorpay-risk-agent.streamlit.app/)

---


## Key Features

- **Interpretable Synthetic Data Generation**: Generates synthetic transaction data (~600 rows) with realistic domain features (`amount_deviation_ratio`, `is_night`, `velocity_last_hour`, `location_mismatch`, `device_change`, `customer_tenure_days`) and ~6% label noise rather than anonymized PCA components.
- **RandomForest Fraud Classifier**: Class-balanced model trained with stratified 75/25 split reporting precision, recall, F1 score, and feature importances.
- **Defensive Decision Policy & Safety Gate**:
  - `Probability >= 0.75` $\rightarrow$ `HOLD`
  - `0.40 <= Probability < 0.75` $\rightarrow$ `ESCALATE`
  - `Probability < 0.40` $\rightarrow$ `CLEAR`
  - **Safety Gate**: Tracks running `HOLD` rate across the batch. If the running hold rate exceeds **25%**, the agent automatically downgrades subsequent `HOLD` decisions to `ESCALATE` to prevent catastrophic automated customer lockouts.
- **Explainable Audit Logging**: Identifies top risk-contributing feature for each transaction and exports complete history to `outputs/audit_trail.csv`.
- **Zero-Dependency Unit Test Suite**: Comprehensive testing using Python's standard `unittest` framework.

---

## Project Structure

```text
├── data/
│   └── transactions.csv         # Generated synthetic transaction dataset
├── outputs/
│   └── audit_trail.csv          # Agent evaluation logs & top feature explanations
├── src/
│   ├── __init__.py
│   ├── generate_data.py         # Synthetic transaction dataset generator
│   ├── model.py                 # RandomForest training, evaluation & metrics
│   └── agent.py                 # Auto-responder agent, decision policy & safety gate
├── tests/
│   ├── __init__.py
│   └── test_agent.py            # Standard unittest suite
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation & Setup

1. **Clone / Open Workspace**
   Ensure you are in the project root directory.

2. **Set Up Virtual Environment** (Optional but recommended)
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Generate Synthetic Data
```bash
python src/generate_data.py
```
Outputs dataset to `data/transactions.csv`.

### 2. Train Model & Evaluate Features
```bash
python src/model.py
```
Trains the `RandomForestClassifier`, displays test performance metrics (Precision, Recall, F1), and outputs feature importances.

### 4. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Launches a modern, interactive Streamlit web dashboard for real-time risk decisioning, single transaction analysis, batch dataset evaluation, and audit trail inspection.

---

## 🌐 Web Dashboard Overview (`app.py`)

The Streamlit dashboard provides a fintech risk-monitoring interface built directly on top of the `FraudAgent` core engine:

- **Single Transaction Analysis**: Interactive form allowing real-time parameter input (amount, merchant average, velocity, hour, security flags, customer tenure) to compute fraud risk scores, operational actions (`CLEAR`, `ESCALATE`, `HOLD`), top risk drivers, and safety gate status.
- **Batch Dataset Analysis**: Execute full batch processing across dataset records, view decision breakdown charts, and inspect risk factor frequencies.
- **Audit Trail Inspector**: Search, filter, and inspect generated decision logs from `outputs/audit_trail.csv`.
- **System Architecture & Metrics**: Displays model performance (Precision, Recall, F1) and explains policy threshold rules.


---

## Decision Policy & Safety Gate Architecture

| Fraud Probability | Default Decision | Safety Gate Action (if Hold Rate > 25%) |
| :--- | :--- | :--- |
| **$\ge 0.75$** | `HOLD` | Downgraded to `ESCALATE` |
| **$0.40 - 0.74$** | `ESCALATE` | `ESCALATE` |
| **$< 0.40$** | `CLEAR` | `CLEAR` |

> **Note on Agent Safety:** The agent is strictly defensive and never takes irreversible automated actions (e.g. auto-refund or auto-cancellation). When the safety gate triggers due to high false-positive risk or abnormal batch spikes, it escalates transactions for human review instead of holding transactions automatically.

---

## Empirical Model Performance (Latest Run)

Results on the held-out test evaluation set (150 transactions / 25% stratified split):

| Metric | Score | Percentage | Notes |
| :--- | :--- | :--- | :--- |
| **Precision** | `0.5000` | 50.00% | 1 out of 2 flagged transactions is true fraud |
| **Recall** | `0.4783` | 47.83% | Catches ~48% of true fraud cases |
| **F1 Score** | `0.4889` | 48.89% | Harmonic mean of precision and recall |
| **Accuracy** | `0.8500` | 85.00% | Overall test set classification accuracy |

### Feature Importances Breakdown
1. `amount_deviation_ratio`: **0.2906** (29.1%)
2. `amount`: **0.2647** (26.5%)
3. `merchant_avg_amount`: **0.1051** (10.5%)
4. `customer_tenure_days`: **0.0751** (7.5%)
5. `hour_of_day`: **0.0717** (7.2%)
6. `location_mismatch`: **0.0676** (6.8%)
7. `velocity_last_hour`: **0.0530** (5.3%)
8. `is_night`: **0.0471** (4.7%)
9. `device_change`: **0.0251** (2.5%)

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 17da914 (Add buildathon risk operations dashboard)
### Batch Decision Action Summary (600 Transactions Processed)

| Operational Decision | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **HOLD** | **61** | **10.2%** | High-risk transactions ($\ge 0.75$) held for intervention |
| **ESCALATE** | **62** | **10.3%** | Moderate-risk transactions ($0.40 - 0.74$) routed for step-up verification |
| **CLEAR** | **477** | **79.5%** | Low-risk transactions ($< 0.40$) approved automatically |
| **Total** | **600** | **100.0%** | Full batch evaluated with Safety Gate = `False` (Hold rate 10.2% $< 25\%$) |


<<<<<<< HEAD
=======
>>>>>>> d7459c8 (AI Risk Manager: fraud verifier and auto-responder agent)
=======
>>>>>>> 17da914 (Add buildathon risk operations dashboard)
---

## Development Debugging & What Broke

During implementation and testing, two key operational issues were encountered and resolved:

### 1. Standalone Script Import Failures (`ModuleNotFoundError`)
- **Issue**: Running `python src/agent.py` or `python src/model.py` directly caused Python to fail with `ModuleNotFoundError: No module named 'src'` because the module path wasn't in `sys.path`.
- **Fix**: Added dynamic project root resolution `sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))` at the top of entrypoint scripts, allowing both direct script execution (`python src/agent.py`) and module invocation (`python -m src.agent`).

### 2. Immediate False Positive Safety Gate Activation on Single/Small Batches
- **Issue**: In `test_agent.py`, evaluating a high-risk transaction on batch start calculated a prospective hold rate of `(0 + 1) / (0 + 1) = 100%`. Since `100% > 25%`, the safety gate triggered instantly on transaction #1, locking down all subsequent items before establishing a sample size.
- **Fix**: Updated `FraudAgent.evaluate_transaction` to enforce a minimum **10-transaction warm-up window** (`total_processed >= 10`) before evaluating the 25% running hold rate threshold. This prevents false activations on initial transactions while maintaining rate-limiting protection across the full batch.

