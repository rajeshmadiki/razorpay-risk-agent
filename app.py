import os
import sys
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import existing domain logic without modifying core decision algorithms
try:
    from src.model import load_data, train_fraud_model, FEATURE_COLS
    from src.agent import FraudAgent, run_agent_batch
except Exception as e:
    st.error(f"Initialization Error: Unable to import core risk agent modules. Details: {e}")
    st.stop()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Razorpay Risk Agent - Risk Operations Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RESTRAINED PROFESSIONAL FINTECH THEME & CSS ---
st.markdown(r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500;600;700&display=swap');

    /* --- THEME CSS VARIABLES --- */
    :root, [data-theme="dark"] {
        --bg-canvas: #0d1117;
        --bg-card: #161b22;
        --bg-sidebar: #010409;
        --border-color: #30363d;
        --border-subtle: #21262d;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-heading: #ffffff;
        --accent-blue: #58a6ff;
        --meter-track: #21262d;
        --input-bg: #161b22;
        --input-text: #f0f6fc;
        --input-border: #30363d;
        --chip-bg: #161b22;
        --clear-bg: #0d2d1d;
        --clear-border: #1b472c;
        --clear-text: #3fb950;
        --escalate-bg: #341a00;
        --escalate-border: #543000;
        --escalate-text: #d29922;
        --hold-bg: #3c1118;
        --hold-border: #6e1a24;
        --hold-text: #f85149;
    }

    @media (prefers-color-scheme: light) {
        :root:not([data-theme="dark"]) {
            --bg-canvas: #f8f9fa;
            --bg-card: #ffffff;
            --bg-sidebar: #f1f5f9;
            --border-color: #e2e8f0;
            --border-subtle: #cbd5e1;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-heading: #0f172a;
            --accent-blue: #0284c7;
            --meter-track: #e2e8f0;
            --input-bg: #ffffff;
            --input-text: #0f172a;
            --input-border: #cbd5e1;
            --chip-bg: #f1f5f9;
            --clear-bg: #f0fdf4;
            --clear-border: #bbf7d0;
            --clear-text: #15803d;
            --escalate-bg: #fffbeb;
            --escalate-border: #fef08a;
            --escalate-text: #b45309;
            --hold-bg: #fef2f2;
            --hold-border: #fecaca;
            --hold-text: #b91c1c;
        }
    }

    [data-theme="light"] {
        --bg-canvas: #f8f9fa;
        --bg-card: #ffffff;
        --bg-sidebar: #f1f5f9;
        --border-color: #e2e8f0;
        --border-subtle: #cbd5e1;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-heading: #0f172a;
        --accent-blue: #0284c7;
        --meter-track: #e2e8f0;
        --input-bg: #ffffff;
        --input-text: #0f172a;
        --input-border: #cbd5e1;
        --chip-bg: #f1f5f9;
        --clear-bg: #f0fdf4;
        --clear-border: #bbf7d0;
        --clear-text: #15803d;
        --escalate-bg: #fffbeb;
        --escalate-border: #fef08a;
        --escalate-text: #b45309;
        --hold-bg: #fef2f2;
        --hold-border: #fecaca;
        --hold-text: #b91c1c;
    }

    /* --- BUG 2 FIX: Streamlit Header & Top Viewport Clearance --- */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 99 !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }

    div.block-container {
        padding-top: 3.75rem !important;
        padding-bottom: 2rem !important;
        max-width: 1320px !important;
    }

    /* --- BUG 1 FIX: Canvas & Layout Theme Styling --- */
    .stApp {
        background-color: var(--bg-canvas) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4,
    div[data-testid="stMarkdownContainer"] h5,
    div[data-testid="stMarkdownContainer"] h6 {
        color: var(--text-heading) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span,
    div[data-testid="stMarkdownContainer"] li {
        color: var(--text-primary);
    }

    /* Header Section */
    .header-wrapper {
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 0.75rem;
        margin-bottom: 1.0rem;
        margin-top: 0 !important;
        position: relative;
        z-index: 1;
    }
    .header-tag {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: var(--accent-blue) !important;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-heading) !important;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.1;
    }
    .header-desc {
        font-size: 0.9rem;
        color: var(--text-secondary) !important;
        margin-bottom: 0;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: var(--clear-bg);
        color: var(--clear-text) !important;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid var(--clear-border);
    }
    .status-dot-pulse {
        width: 7px;
        height: 7px;
        background-color: var(--clear-text);
        border-radius: 50%;
        animation: pulse-slow 2.5s infinite ease-in-out;
    }
    @keyframes pulse-slow {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.95); }
        100% { opacity: 1; transform: scale(1); }
    }

    .pulse-dot-red {
        width: 7px;
        height: 7px;
        background-color: var(--hold-text);
        border-radius: 50%;
        animation: pulse-red 2.0s infinite ease-in-out;
    }
    @keyframes pulse-red {
        0% { opacity: 1; }
        50% { opacity: 0.35; }
        100% { opacity: 1; }
    }

    .pulse-dot-amber {
        width: 7px;
        height: 7px;
        background-color: var(--escalate-text);
        border-radius: 50%;
        animation: pulse-amber 2.5s infinite ease-in-out;
    }
    @keyframes pulse-amber {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }

    .dot-stable-green {
        width: 7px;
        height: 7px;
        background-color: var(--clear-text);
        border-radius: 50%;
    }

    /* Header Fact Chips */
    .fact-chip {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-secondary);
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        padding: 2px 8px;
        border-radius: 4px;
    }

    /* Section Title */
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-heading) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
        padding-bottom: 4px;
        border-bottom: 1px solid var(--border-color);
    }

    /* Input Controls Theme Adaptation */
    div[data-testid="stInputInstruction"] {
        position: relative !important;
        display: block !important;
        margin-top: 4px !important;
        font-size: 0.72rem !important;
        color: var(--text-secondary) !important;
        clear: both !important;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div {
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 6px !important;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.84rem !important;
    }

    /* Form and Expander styling */
    div[data-testid="stForm"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--text-heading) !important;
    }
    div[data-testid="stExpander"] summary span {
        color: var(--text-heading) !important;
    }
    div[data-testid="stDataFrame"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
    }

    /* Semantic Decision Banners */
    .decision-banner {
        border-radius: 6px;
        padding: 14px 18px;
        text-align: center;
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        font-size: 1.35rem;
        letter-spacing: 0.03em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .banner-clear {
        background-color: var(--clear-bg);
        color: var(--clear-text) !important;
        border: 1px solid var(--clear-border);
    }
    .banner-escalate {
        background-color: var(--escalate-bg);
        color: var(--escalate-text) !important;
        border: 1px solid var(--escalate-border);
    }
    .banner-hold {
        background-color: var(--hold-bg);
        color: var(--hold-text) !important;
        border: 1px solid var(--hold-border);
    }

    /* Metric Tiles Grid */
    .metric-tile {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 10px 8px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-heading) !important;
    }
    .metric-label {
        font-size: 0.7rem;
        color: var(--text-secondary) !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Decision Pipeline Flowchart Nodes */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 14px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.74rem;
    }
    .pipeline-node {
        color: var(--text-primary) !important;
        padding: 3px 8px;
        border-radius: 4px;
        background: var(--bg-canvas);
        border: 1px solid var(--border-subtle);
        font-weight: 500;
    }
    .pipeline-arrow {
        color: var(--text-secondary) !important;
        font-weight: bold;
    }

    /* Code Blocks & Flowchart Pre-formatted Diagrams Universal Theme Safety */
    code, pre, .stCodeBlock, div[data-testid="stCodeBlock"],
    div[data-testid="stMarkdownContainer"] pre,
    div[data-testid="stMarkdownContainer"] code,
    div[data-testid="stMarkdownContainer"] pre *,
    div[data-testid="stMarkdownContainer"] code *,
    div[data-testid="stCodeBlock"] *,
    .stCodeBlock *, code *, pre * {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }



    /* Custom Meter Styling */
    .meter-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
    }
    .meter-bar-track {
        height: 10px;
        background: var(--meter-track);
        border-radius: 4px;
        position: relative;
        overflow: hidden;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .meter-regions {
        display: flex;
        height: 100%;
        width: 100%;
    }
    .region-low { width: 40%; background: rgba(46, 160, 67, 0.25); border-right: 1px solid var(--border-color); }
    .region-med { width: 35%; background: rgba(210, 153, 34, 0.25); border-right: 1px solid var(--border-color); }
    .region-high { width: 25%; background: rgba(248, 81, 73, 0.25); }

    .meter-pointer-line {
        position: absolute;
        top: 0;
        bottom: 0;
        width: 3px;
        background: var(--text-heading);
        box-shadow: 0 0 4px var(--text-heading);
    }
    .meter-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: var(--text-secondary);
        font-family: 'Roboto Mono', monospace;
    }

    /* Factor Row styling */
    .factor-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        color: var(--text-primary);
        font-family: 'Roboto Mono', monospace;
        margin-bottom: 2px;
    }
    .factor-name {
        color: var(--text-primary);
    }

    /* Primary Action Buttons */
    div.stButton > button {
        background-color: #238636 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid #2ea043 !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        background-color: #2ea043 !important;
    }

    /* Responsive Breakpoints */
    @media (max-width: 768px) {
        div.block-container {
            padding-top: 4.25rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .header-title {
            font-size: 1.6rem !important;
        }
        .pipeline-container {
            flex-wrap: wrap;
            gap: 4px;
            font-size: 0.68rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_risk_engine():
    """
    Load data, train model, and instantiate FraudAgent using core python implementation.
    """
    data_path = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
    df = load_data(data_path)
    model, metrics, feat_importances, (X_train, X_test, y_train, y_test) = train_fraud_model(df)

    agent = FraudAgent(
        model=model,
        feature_importances=feat_importances,
        feature_means=X_train.mean(),
        feature_stds=X_train.std()
    )
    return agent, df, metrics, feat_importances

# Initialize Session State for Recent Activity & Demo Scenario Controls
if "session_activity" not in st.session_state:
    st.session_state["session_activity"] = []

if "demo_input" not in st.session_state:
    st.session_state["demo_input"] = {
        "txn_id": "TXN_9999",
        "amount": 350.0,
        "merchant_avg": 50.0,
        "hour_of_day": 2,
        "velocity": 5,
        "location_mismatch": "Yes",
        "device_change": "Yes",
        "tenure_days": 15
    }

# Initialize Engine with graceful error handling
try:
    agent, df_dataset, model_metrics, feature_importances = initialize_risk_engine()
except Exception as err:
    st.error(f"Engine Load Error: Could not initialize risk engine. Details: {err}")
    st.stop()

# --- SIDEBAR NAVIGATION WITH WORKSPACE PURPOSES ---
st.sidebar.title("🛡️ Risk Console")
page = st.sidebar.radio(
    "Workspaces",
    [
        "Dashboard & Single Transaction",
        "Batch Analysis",
        "Audit Trail",
        "About & Architecture"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Local Risk Engine Status")
st.sidebar.markdown('<div class="status-badge"><div class="status-dot-pulse"></div>LOCAL RISK ENGINE ONLINE</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="margin-top: 8px; font-size: 0.75rem; color: var(--text-secondary);">
  • <b>MODEL:</b> RandomForest<br>
  • <b>ENGINE:</b> Local Python Engine<br>
  • <b>MODE:</b> Defense-Only Decisioning<br>
  • <b>DATASET:</b> 600 rows (15.5% fraud)
</div>
""", unsafe_allow_html=True)

# --- MINIMALIST FINTECH GLOBAL HEADER ---
st.markdown('''
<div class="header-wrapper">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <div class="header-tag">RISK OPERATIONS CONSOLE</div>
            <div class="header-title">Razorpay Risk Agent</div>
            <div class="header-desc">Fraud Verification & Automated Operational Risk Decisioning</div>
        </div>
        <div style="text-align: right;">
            <div class="status-badge"><div class="status-dot-pulse"></div>LOCAL RISK ENGINE ONLINE</div>
            <div style="margin-top: 6px; display: flex; gap: 4px; justify-content: flex-end;">
                <span class="fact-chip">MODEL: RandomForest</span>
                <span class="fact-chip">MODE: Defense-Only</span>
            </div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

# --- WORKSPACE 1: DASHBOARD & SINGLE TRANSACTION ANALYSIS ---
if page == "Dashboard & Single Transaction":

    # --- DEMO SCENARIOS PRE-FILL HELPERS ---
    st.markdown('<div class="section-title">⚡ Demo Scenarios (Input Helpers)</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        if st.button("Load Clear Scenario (Low Risk)", width="stretch"):
            st.session_state["demo_input"] = {
                "txn_id": "TXN_CLEAR_01",
                "amount": 25.0,
                "merchant_avg": 50.0,
                "hour_of_day": 14,
                "velocity": 1,
                "location_mismatch": "No",
                "device_change": "No",
                "tenure_days": 600
            }
    with sc2:
        if st.button("Load Escalate Scenario (Medium Risk)", width="stretch"):
            st.session_state["demo_input"] = {
                "txn_id": "TXN_ESC_02",
                "amount": 450.0,
                "merchant_avg": 50.0,
                "hour_of_day": 1,
                "velocity": 4,
                "location_mismatch": "Yes",
                "device_change": "No",
                "tenure_days": 45
            }
    with sc3:
        if st.button("Load Hold Scenario (High Risk)", width="stretch"):
            st.session_state["demo_input"] = {
                "txn_id": "TXN_HOLD_03",
                "amount": 1280.0,
                "merchant_avg": 200.0,
                "hour_of_day": 3,
                "velocity": 8,
                "location_mismatch": "Yes",
                "device_change": "Yes",
                "tenure_days": 15
            }

    col_input, col_result = st.columns([1.15, 1.0], gap="large")

    demo_vals = st.session_state["demo_input"]

    with col_input:
        st.markdown('<div class="section-title">Transaction Signals Input</div>', unsafe_allow_html=True)

        with st.form("transaction_form"):
            st.markdown("##### Identity")
            txn_id = st.text_input("Transaction ID", value=demo_vals["txn_id"], help="Unique payment transaction reference")

            st.markdown("##### Financial Signals")
            f1, f2 = st.columns(2)
            with f1:
                amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=50000.0, value=float(demo_vals["amount"]), step=10.0, help="Attempted charge value")
            with f2:
                merchant_avg = st.number_input("Merchant Average Amount ($)", min_value=1.0, max_value=10000.0, value=float(demo_vals["merchant_avg"]), step=5.0, help="Baseline average transaction amount for this merchant")

            st.markdown("##### Behavioral & Security Signals")
            b1, b2 = st.columns(2)
            with b1:
                hour_of_day = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=int(demo_vals["hour_of_day"]), help="Local hour of payment attempt")
                velocity = st.number_input("Velocity (last 1 hr)", min_value=1, max_value=20, value=int(demo_vals["velocity"]), help="Number of attempts from account in past hour")
            with b2:
                location_mismatch = st.selectbox("Location Mismatch", options=["No", "Yes"], index=1 if demo_vals["location_mismatch"] == "Yes" else 0, help="Flag if IP location differs from billing country")
                device_change = st.selectbox("Device Change Detected", options=["No", "Yes"], index=1 if demo_vals["device_change"] == "Yes" else 0, help="Flag if payment originates from a new device fingerprint")

            tenure_days = st.number_input("Customer Tenure (days)", min_value=1, max_value=2000, value=int(demo_vals["tenure_days"]), help="Account age in days")

            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("Analyze Transaction Risk", width="stretch")

    # Derived features calculation (strictly adhering to model inputs)
    calc_deviation_ratio = round(amount / merchant_avg if merchant_avg > 0 else 1.0, 2)
    calc_is_night = 1 if hour_of_day in [22, 23, 0, 1, 2, 3, 4, 5] else 0
    calc_loc_mismatch = 1 if location_mismatch == "Yes" else 0
    calc_dev_change = 1 if device_change == "Yes" else 0

    input_row = pd.Series({
        "transaction_id": txn_id,
        "amount": amount,
        "merchant_avg_amount": merchant_avg,
        "amount_deviation_ratio": calc_deviation_ratio,
        "hour_of_day": hour_of_day,
        "is_night": calc_is_night,
        "velocity_last_hour": velocity,
        "location_mismatch": calc_loc_mismatch,
        "device_change": calc_dev_change,
        "customer_tenure_days": tenure_days
    })

    # Evaluate transaction using real FraudAgent engine
    if submit_btn:
        with st.spinner("Executing risk engine evaluation..."):
            prob, decision, top_feature = agent.evaluate_transaction(input_row)

            eval_time = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state["session_activity"].insert(0, {
                "Transaction ID": txn_id,
                "Amount ($)": f"${amount:,.2f}",
                "Fraud Probability": f"{prob * 100:.1f}%",
                "Decision": decision,
                "Time": eval_time
            })
    else:
        prob, decision, top_feature = agent.evaluate_transaction(input_row)

    with col_result:
        st.markdown('<div class="section-title">Risk Decision Result</div>', unsafe_allow_html=True)

        # Decision Banner & Semantic Indicator
        if decision == "CLEAR":
            banner_class = "banner-clear"
            banner_label = '<div class="dot-stable-green"></div> VERIFIED — CLEAR'
            risk_cat = "LOW RISK"
            risk_color = "var(--clear-text)"
            node_style_clear = "background:var(--clear-bg); color:var(--clear-text); border:1px solid var(--clear-border);"
            node_style_esc = "color:var(--text-primary);"
            node_style_hold = "color:var(--text-primary);"
        elif decision == "ESCALATE":
            banner_class = "banner-escalate"
            banner_label = '<div class="pulse-dot-amber"></div> REVIEW REQUIRED — ESCALATE'
            risk_cat = "MEDIUM RISK"
            risk_color = "var(--escalate-text)"
            node_style_clear = "color:var(--text-primary);"
            node_style_esc = "background:var(--escalate-bg); color:var(--escalate-text); border:1px solid var(--escalate-border);"
            node_style_hold = "color:var(--text-primary);"
        else:
            banner_class = "banner-hold"
            banner_label = '<div class="pulse-dot-red"></div> HIGH RISK — HOLD'
            risk_cat = "HIGH RISK"
            risk_color = "var(--hold-text)"
            node_style_clear = "color:var(--text-primary);"
            node_style_esc = "color:var(--text-primary);"
            node_style_hold = "background:var(--hold-bg); color:var(--hold-text); border:1px solid var(--hold-border);"


        st.markdown(f'<div class="decision-banner {banner_class}">{banner_label}</div>', unsafe_allow_html=True)

        # Metric Grid Tiles
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-tile"><div class="metric-value" style="color:{risk_color}">{prob * 100:.1f}%</div><div class="metric-label">Fraud Probability</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{decision}</div><div class="metric-label">Policy Decision</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-tile"><div class="metric-value" style="color:{risk_color}">{risk_cat}</div><div class="metric-label">Risk Category</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # PROBABILITY RISK METER WITH POINTER & POLICY REGIONS
        pointer_pct = round(prob * 100.0, 1)
        st.markdown(f"""
        <div class="meter-container">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size:0.75rem; color:var(--text-secondary); font-weight:600; font-family:'Roboto Mono',monospace;">PROBABILITY RISK METER</span>
                <span style="font-family:'Roboto Mono',monospace; font-size:1.1rem; font-weight:700; color:{risk_color}; margin-left:auto;">{pointer_pct:.1f}%</span>
            </div>
            <div class="meter-bar-track">
                <div class="meter-regions">
                    <div class="region-low"></div>
                    <div class="region-med"></div>
                    <div class="region-high"></div>
                </div>
                <div class="meter-pointer-line" style="left: calc({pointer_pct}% - 1px);"></div>
            </div>
            <div class="meter-labels">
                <span>0% (LOW: CLEAR &lt; 40%)</span>
                <span>40% (MED: ESCALATE)</span>
                <span>75% (HIGH: HOLD &ge; 75%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # COMPACT DECISION PIPELINE FLOWCHART
        st.markdown(f"""
        <div class="pipeline-container">
            <span class="pipeline-node">01 INPUT</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-node">02 FEATURES</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-node">03 MODEL</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-node">04 PROB ({pointer_pct:.0f}%)</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-node">05 POLICY</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-node" style="{node_style_clear if decision=='CLEAR' else (node_style_esc if decision=='ESCALATE' else node_style_hold)}">06 {decision}</span>
        </div>
        """, unsafe_allow_html=True)

        # RISK INTELLIGENCE FACTORS BREAKDOWN
        st.markdown('<div class="section-title">Risk Intelligence Factors</div>', unsafe_allow_html=True)
        st.info(f"**Highest-Weighted Model Signal:** `{top_feature}`")

        dev_risk = min(calc_deviation_ratio / 8.0, 1.0)
        vel_risk = min(velocity / 12.0, 1.0)
        loc_risk = 0.85 if calc_loc_mismatch else 0.15
        dev_c_risk = 0.75 if calc_dev_change else 0.10

        st.markdown(f"""
        <div style="margin-bottom: 8px;">
            <div class="factor-row"><span class="factor-name">01 Amount Deviation Ratio ({calc_deviation_ratio}x)</span><span class="factor-val" style="color:var(--accent-blue)">{'HIGH' if dev_risk > 0.6 else 'NORMAL'}</span></div>
            <div style="background:var(--meter-track); height:5px; border-radius:3px;"><div style="background:var(--accent-blue); width:{dev_risk*100}%; height:5px; border-radius:3px;"></div></div>
        </div>
        <div style="margin-bottom: 8px;">
            <div class="factor-row"><span class="factor-name">02 Velocity ({velocity} txns/hr)</span><span class="factor-val" style="color:var(--escalate-text)">{'HIGH' if vel_risk > 0.5 else 'NORMAL'}</span></div>
            <div style="background:var(--meter-track); height:5px; border-radius:3px;"><div style="background:var(--escalate-text); width:{vel_risk*100}%; height:5px; border-radius:3px;"></div></div>
        </div>
        <div style="margin-bottom: 8px;">
            <div class="factor-row"><span class="factor-name">03 Device & Location Security</span><span class="factor-val" style="color:var(--hold-text)">{'FLAGGED' if (calc_loc_mismatch or calc_dev_change) else 'NORMAL'}</span></div>
            <div style="background:var(--meter-track); height:5px; border-radius:3px;"><div style="background:var(--hold-text); width:{max(loc_risk, dev_c_risk)*100}%; height:5px; border-radius:3px;"></div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("❓ Detailed Decision Explanation", expanded=False):
            st.markdown(rf"""
            - **Primary Risk Factor:** `{top_feature}` drove the highest feature importance weight.
            - **Amount Deviation:** Purchase is `{calc_deviation_ratio}x` merchant baseline (${merchant_avg}).
            - **Night Purchase Window:** `{'Yes (' + str(hour_of_day) + ':00)' if calc_is_night else 'No'}`.
            - **Model Evaluation:** `RandomForest` predicted raw fraud probability at `{prob * 100:.2f}%`.
            - **Policy Decision:** Assigned **`{decision}`** based on thresholds (<40% CLEAR, 40-75% ESCALATE, >=75% HOLD).
            - **Safety Control Gate:** `{'ACTIVE (Downgrading HOLDs to ESCALATE)' if agent.gate_triggered else 'NORMAL (Running hold rate within limits)'}`.
            """)

    # --- RECENT DECISIONS ACTIVITY ---
    st.markdown("---")
    st.markdown('<div class="section-title">Recent Session Activity</div>', unsafe_allow_html=True)
    if st.session_state["session_activity"]:
        act_df = pd.DataFrame(st.session_state["session_activity"])
        st.dataframe(act_df, width="stretch", height=160)
    else:
        st.caption("No real-time evaluations submitted in current session yet. Submit a transaction above to record activity.")

# --- WORKSPACE 2: BATCH ANALYSIS ---
elif page == "Batch Analysis":
    st.markdown("### Batch Transaction Verification Console")
    st.write("Execute population-level batch risk evaluation across `data/transactions.csv` using `run_agent_batch()`.")

    if st.button("Process Full Batch Dataset", width="stretch"):
        with st.spinner("Processing batch transactions and evaluating safety gate limits..."):
            try:
                data_path = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
                output_path = os.path.join(PROJECT_ROOT, "outputs", "audit_trail.csv")
                batch_agent, audit_df = run_agent_batch(data_path, output_path)
                st.session_state["audit_df"] = audit_df
                st.session_state["batch_agent"] = batch_agent
                st.success("Batch evaluation completed! Results saved to `outputs/audit_trail.csv`.")
            except Exception as batch_err:
                st.error(f"Error during batch execution: {batch_err}")

    # Load audit trail if available
    audit_file = os.path.join(PROJECT_ROOT, "outputs", "audit_trail.csv")
    if os.path.exists(audit_file):
        audit_df = pd.read_csv(audit_file)

        total_txns = len(audit_df)
        holds = (audit_df["decision"] == "HOLD").sum()
        escalates = (audit_df["decision"] == "ESCALATE").sum()
        clears = (audit_df["decision"] == "CLEAR").sum()

        st.markdown("<br>", unsafe_allow_html=True)

        # Factual Summary Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{total_txns}</div><div class="metric-label">Total Processed</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-tile"><div class="metric-value" style="color:var(--clear-text)">{clears}</div><div class="metric-label">CLEAR ({clears/total_txns*100:.1f}%)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-tile"><div class="metric-value" style="color:var(--escalate-text)">{escalates}</div><div class="metric-label">ESCALATE ({escalates/total_txns*100:.1f}%)</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-tile"><div class="metric-value" style="color:var(--hold-text)">{holds}</div><div class="metric-label">HOLD ({holds/total_txns*100:.1f}%)</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Horizontal Action Distribution Bar
        hold_pct = (holds / total_txns) * 100.0
        esc_pct = (escalates / total_txns) * 100.0
        clear_pct = (clears / total_txns) * 100.0

        st.markdown("##### Population Action Distribution Breakdown")
        st.markdown(f"""
        <div style="background:var(--bg-card); border:1px solid var(--border-color); border-radius:6px; padding:12px; margin-bottom:16px;">
            <div style="display:flex; height:12px; border-radius:4px; overflow:hidden; margin-bottom:8px;">
                <div style="width:{clear_pct}%; background:var(--clear-text);" title="CLEAR"></div>
                <div style="width:{esc_pct}%; background:var(--escalate-text);" title="ESCALATE"></div>
                <div style="width:{hold_pct}%; background:var(--hold-text);" title="HOLD"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-family:'Roboto Mono',monospace; font-size:0.75rem; color:var(--text-secondary);">
                <span style="color:var(--clear-text);">■ CLEAR: {clears} ({clear_pct:.1f}%)</span>
                <span style="color:var(--escalate-text);">■ ESCALATE: {escalates} ({esc_pct:.1f}%)</span>
                <span style="color:var(--hold-text);">■ HOLD: {holds} ({hold_pct:.1f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Operational Risk Controls & Safety Gate Status
        st.markdown('<div class="section-title">Risk Control & Safety Gate</div>', unsafe_allow_html=True)
        gate_triggered = (holds / total_txns) > 0.25
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{"TRIGGERED" if gate_triggered else "WITHIN LIMIT"}</div><div class="metric-label">Safety Gate Status</div></div>', unsafe_allow_html=True)
        with g2:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{hold_pct:.1f}%</div><div class="metric-label">Running Hold Rate</div></div>', unsafe_allow_html=True)
        with g3:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">25.0%</div><div class="metric-label">Max Hold Threshold</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.markdown("#### Primary Risk Signal Frequencies")
            feat_df = audit_df["top_contributing_feature"].value_counts().reset_index()
            feat_df.columns = ["Feature Name", "Flag Count"]
            st.dataframe(feat_df, width="stretch")

        with col_chart2:
            st.markdown("#### Held-out Model Evaluation Evidence")
            st.markdown(f"""
            - **Precision**: `{model_metrics['precision']:.4f}` ({model_metrics['precision']*100:.1f}%)
            - **Recall**: `{model_metrics['recall']:.4f}` ({model_metrics['recall']*100:.1f}%)
            - **F1 Score**: `{model_metrics['f1']:.4f}` ({model_metrics['f1']*100:.1f}%)
            - **Evaluation Split**: `25% Stratified Test Split`
            """)

    else:
        st.info("No batch audit trail found. Click the button above to run batch evaluation.")

# --- WORKSPACE 3: AUDIT TRAIL ---
elif page == "Audit Trail":
    st.markdown("### Operational Audit Trail Log")
    st.caption("AUDITABLE DECISION LOG — Traceable compliance record saved to outputs/audit_trail.csv")

    audit_file = os.path.join(PROJECT_ROOT, "outputs", "audit_trail.csv")
    if os.path.exists(audit_file):
        audit_df = pd.read_csv(audit_file)

        # Filters
        f1, f2 = st.columns(2)
        with f1:
            selected_decision = st.multiselect("Filter by Decision Category", options=["CLEAR", "ESCALATE", "HOLD"], default=["CLEAR", "ESCALATE", "HOLD"])
        with f2:
            search_id = st.text_input("Search Transaction ID", value="")

        filtered_df = audit_df[audit_df["decision"].isin(selected_decision)]
        if search_id.strip():
            filtered_df = filtered_df[filtered_df["transaction_id"].str.contains(search_id.strip(), case=False)]

        st.dataframe(filtered_df, width="stretch", height=480)
        st.caption(f"Displaying {len(filtered_df)} of {len(audit_df)} total log records.")
    else:
        st.warning("Audit trail log file not found. Execute a batch evaluation to generate records.")

# --- WORKSPACE 4: ABOUT & ARCHITECTURE ---
elif page == "About & Architecture":
    st.markdown("### System Architecture & Risk Pipeline")

    st.markdown(r"""
    ### Architecture Pipeline Diagram

    ```text
    [ Transaction Signals Input ]
                 │
                 ▼
    [ Feature Extraction & Normalization ]
                 │
                 ▼
    [ RandomForest Classifier Engine ]
                 │
                 ▼
    [ Continuous Fraud Probability Score ]
                 │
                 ▼
    [ Defensive Safety Control (25% Limit) ]
                 │
                 ▼
    [ Policy Decision Engine: CLEAR / ESCALATE / HOLD ]
                 │
                 ▼
    [ Audit Log Export (outputs/audit_trail.csv) ]
    ```

    ### Pipeline Stage Explanations
    1. **Transaction Signals Input**: Ingests payment attributes (`amount`, `merchant_avg_amount`, `velocity`, `hour_of_day`, `location_mismatch`, `device_change`, `customer_tenure_days`).
    2. **Feature Extraction**: Computes derived indicators (`amount_deviation_ratio`, `is_night`).
    3. **RandomForest Engine**: Predicts raw risk probability with class balancing on a 75/25 stratified split.
    4. **Defensive Safety Control**: Evaluates probability threshold ($<40\%$ CLEAR, $40-75\%$ ESCALATE, $\ge 75\%$ HOLD). If running hold rate $> 25\%$, downgrades `HOLD` to `ESCALATE`.
    5. **Audit Log Export**: Exports decision log with ISO timestamp and primary risk driver to disk.
    """)

    st.markdown("---")
    st.markdown("##### Defensive Risk Decisioning — Friction vs. Fraud Trade-Off")
    st.markdown("""
    - **False Positive (Legitimate transaction incorrectly held/escalated)**: Business impact depends on transaction value and review/decline policy. Controlled via 3-tier policy and 25% safety rate-limiting gate.
    - **False Negative (Fraudulent transaction incorrectly cleared)**: Causes direct monetary chargeback loss. Controlled via RandomForest class weighting and high-risk thresholding.
    """)

    st.markdown("---")
    st.markdown("##### Measured Model Evaluation Metrics (Held-Out Test Set)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Precision", f"{model_metrics['precision']:.4f}")
    m_col2.metric("Recall", f"{model_metrics['recall']:.4f}")
    m_col3.metric("F1 Score", f"{model_metrics['f1']:.4f}")
    m_col4.metric("Evaluation Split", "25% Stratified")
