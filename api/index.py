import os
import sys
import json
import pandas as pd
import numpy as np
from http.server import BaseHTTPRequestHandler

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.model import load_data, train_fraud_model, FEATURE_COLS
    from src.agent import FraudAgent, run_agent_batch
except Exception as e:
    pass

# Global model initialization cache
_agent_cache = None
_model_metrics_cache = None

def get_risk_agent():
    global _agent_cache, _model_metrics_cache
    if _agent_cache is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
        df = load_data(data_path)
        model, metrics, feat_importances, (X_train, X_test, y_train, y_test) = train_fraud_model(df)
        _model_metrics_cache = metrics
        _agent_cache = FraudAgent(
            model=model,
            feature_importances=feat_importances,
            feature_means=X_train.mean(),
            feature_stds=X_train.std()
        )
    return _agent_cache, _model_metrics_cache

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            amount = float(data.get("amount", 350.0))
            merchant_avg = float(data.get("merchant_avg_amount", 50.0))
            dev_ratio = round(amount / merchant_avg if merchant_avg > 0 else 1.0, 2)
            hour = int(data.get("hour_of_day", 2))
            is_night = 1 if hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
            velocity = int(data.get("velocity_last_hour", 5))
            loc_mismatch = 1 if data.get("location_mismatch") in ["Yes", 1] else 0
            dev_change = 1 if data.get("device_change") in ["Yes", 1] else 0
            tenure = int(data.get("customer_tenure_days", 15))

            row = pd.Series({
                "transaction_id": data.get("transaction_id", "TXN_9999"),
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

            agent, metrics = get_risk_agent()
            prob, decision, top_feature = agent.evaluate_transaction(row)

            response_payload = {
                "fraud_probability": round(prob, 4),
                "decision": decision,
                "top_contributing_feature": top_feature,
                "amount_deviation_ratio": dev_ratio,
                "is_night": is_night,
                "safety_gate_triggered": agent.gate_triggered
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception as err:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(err)}).encode('utf-8'))

    def do_GET(self):
        if "/api/metrics" in self.path:
            agent, metrics = get_risk_agent()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(metrics).encode('utf-8'))
            return

        # Serve Python-generated Risk Operations Console HTML
        agent, metrics = get_risk_agent()
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Razorpay Risk Agent - Vercel Python Engine</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 antialiased p-4 min-h-screen">
  <div class="max-w-6xl mx-auto space-y-4">
    <div class="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-3 gap-3">
      <div>
        <div class="text-xs font-mono font-bold tracking-widest text-sky-400 uppercase">RISK OPERATIONS CONSOLE</div>
        <h1 class="text-2xl font-extrabold text-white tracking-tight">Razorpay Risk Agent</h1>
        <p class="text-xs text-slate-400">Powered by Python Serverless Machine Learning Engine on Vercel</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="inline-flex items-center gap-2 bg-emerald-950/80 text-emerald-400 text-xs font-mono font-semibold px-3 py-1.5 rounded-md border border-emerald-800">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          PYTHON MODEL ENGINE ONLINE
        </span>
      </div>
    </div>

    <div class="grid grid-cols-3 gap-3">
      <button onclick="loadScenario('clear')" class="bg-emerald-950/40 hover:bg-emerald-900/50 text-emerald-400 text-xs font-mono font-semibold py-2 px-3 rounded-lg border border-emerald-900 transition text-center">⚡ Load CLEAR Scenario</button>
      <button onclick="loadScenario('escalate')" class="bg-amber-950/40 hover:bg-amber-900/50 text-amber-400 text-xs font-mono font-semibold py-2 px-3 rounded-lg border border-amber-900 transition text-center">⚠️ Load ESCALATE Scenario</button>
      <button onclick="loadScenario('hold')" class="bg-rose-950/40 hover:bg-rose-900/50 text-rose-400 text-xs font-mono font-semibold py-2 px-3 rounded-lg border border-rose-900 transition text-center">⛔ Load HOLD Scenario</button>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-12 gap-5">
      <div class="lg:col-span-6 bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
        <div class="text-xs font-mono font-bold text-slate-300 uppercase border-b border-slate-800 pb-2">💳 Transaction Inputs</div>
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Transaction ID</label>
          <input type="text" id="txn_id" value="TXN_9999" class="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100">
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Amount ($)</label>
            <input type="number" id="amount" value="350" class="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100">
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Merchant Avg ($)</label>
            <input type="number" id="merchant_avg" value="50" class="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100">
          </div>
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Hour of Day (0-23)</label>
            <input type="number" id="hour_of_day" min="0" max="23" value="2" class="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100">
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Velocity (last 1hr)</label>
            <input type="number" id="velocity" value="5" class="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100">
          </div>
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Location Mismatch</label>
            <select id="location_mismatch" class="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100">
              <option value="No">No</option>
              <option value="Yes" selected>Yes</option>
            </select>
          </div>
          <div>
            <label class="block text-xs font-medium text-slate-400 mb-1">Device Change</label>
            <select id="device_change" class="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100">
              <option value="No">No</option>
              <option value="Yes" selected>Yes</option>
            </select>
          </div>
        </div>
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Customer Tenure (days)</label>
          <input type="number" id="tenure_days" value="15" class="w-full bg-slate-950 border border-slate-700 rounded-md px-3 py-1.5 text-xs text-slate-100">
        </div>
        <button onclick="evaluatePython()" class="w-full bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs py-2.5 rounded-lg transition shadow-md">
          ⚡ EVALUATE WITH PYTHON MODEL
        </button>
      </div>

      <div class="lg:col-span-6 bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
        <div class="text-xs font-mono font-bold text-slate-300 uppercase border-b border-slate-800 pb-2">🎯 Python Model Result</div>
        <div id="decision_banner" class="rounded-lg p-3.5 text-center font-mono font-extrabold text-lg border transition bg-emerald-950/80 text-emerald-400 border-emerald-800">
          ● VERIFIED — CLEAR
        </div>
        <div class="grid grid-cols-3 gap-2 text-center">
          <div class="bg-slate-950 border border-slate-800 p-2.5 rounded-lg">
            <div id="prob_metric" class="text-lg font-mono font-bold text-white">14.2%</div>
            <div class="text-[10px] text-slate-400 font-semibold uppercase">Fraud Prob</div>
          </div>
          <div class="bg-slate-950 border border-slate-800 p-2.5 rounded-lg">
            <div id="decision_metric" class="text-lg font-mono font-bold text-white">CLEAR</div>
            <div class="text-[10px] text-slate-400 font-semibold uppercase">Decision</div>
          </div>
          <div class="bg-slate-950 border border-slate-800 p-2.5 rounded-lg">
            <div id="category_metric" class="text-lg font-mono font-bold text-white">LOW RISK</div>
            <div class="text-[10px] text-slate-400 font-semibold uppercase">Risk Level</div>
          </div>
        </div>
        <div class="space-y-1">
          <div class="flex justify-between text-[11px] font-mono text-slate-400 font-semibold">
            <span>PROBABILITY RISK METER</span>
            <span id="meter_pct" class="text-sky-400">14.2%</span>
          </div>
          <div class="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden relative border border-slate-800">
            <div id="meter_bar" class="h-full bg-sky-500 transition-all duration-300" style="width: 14.2%"></div>
          </div>
        </div>
        <div class="space-y-2 pt-2 border-t border-slate-800">
          <div class="text-xs font-mono font-bold text-slate-300">📊 Python Model Intelligence</div>
          <div class="text-xs text-slate-300">Primary Risk Driver: <span id="top_feat" class="font-mono text-sky-400">hour_of_day</span></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function evaluatePython() {{
      let payload = {{
        transaction_id: document.getElementById('txn_id').value,
        amount: parseFloat(document.getElementById('amount').value),
        merchant_avg_amount: parseFloat(document.getElementById('merchant_avg').value),
        hour_of_day: parseInt(document.getElementById('hour_of_day').value),
        velocity_last_hour: parseInt(document.getElementById('velocity').value),
        location_mismatch: document.getElementById('location_mismatch').value,
        device_change: document.getElementById('device_change').value,
        customer_tenure_days: parseInt(document.getElementById('tenure_days').value)
      }};

      try {{
        let res = await fetch('/api/evaluate', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        let data = await res.json();
        let pct = (data.fraud_probability * 100).toFixed(1) + "%";

        document.getElementById('prob_metric').innerText = pct;
        document.getElementById('decision_metric').innerText = data.decision;
        document.getElementById('meter_pct').innerText = pct;
        document.getElementById('meter_bar').style.width = pct;
        document.getElementById('top_feat').innerText = data.top_contributing_feature;

        let b = document.getElementById('decision_banner');
        if (data.decision === 'HOLD') {{
          b.className = "rounded-lg p-3.5 text-center font-mono font-extrabold text-sm border transition bg-rose-950/80 text-rose-400 border-rose-800 animate-pulse";
          b.innerText = "● HIGH RISK — HOLD";
          document.getElementById('category_metric').innerText = "HIGH RISK";
        }} else if (data.decision === 'ESCALATE') {{
          b.className = "rounded-lg p-3.5 text-center font-mono font-extrabold text-sm border transition bg-amber-950/80 text-amber-400 border-amber-800";
          b.innerText = "● REVIEW REQUIRED — ESCALATE";
          document.getElementById('category_metric').innerText = "MEDIUM RISK";
        }} else {{
          b.className = "rounded-lg p-3.5 text-center font-mono font-extrabold text-sm border transition bg-emerald-950/80 text-emerald-400 border-emerald-800";
          b.innerText = "● VERIFIED — CLEAR";
          document.getElementById('category_metric').innerText = "LOW RISK";
        }}
      }} catch (err) {{
        alert("Error invoking Python model API: " + err);
      }}
    }}

    function loadScenario(t) {{
      if (t === 'clear') {{
        document.getElementById('txn_id').value = "TXN_CLEAR_01";
        document.getElementById('amount').value = 25;
        document.getElementById('merchant_avg').value = 50;
        document.getElementById('hour_of_day').value = 14;
        document.getElementById('velocity').value = 1;
        document.getElementById('location_mismatch').value = "No";
        document.getElementById('device_change').value = "No";
        document.getElementById('tenure_days').value = 600;
      }} else if (t === 'escalate') {{
        document.getElementById('txn_id').value = "TXN_ESC_02";
        document.getElementById('amount').value = 450;
        document.getElementById('merchant_avg').value = 50;
        document.getElementById('hour_of_day').value = 1;
        document.getElementById('velocity').value = 4;
        document.getElementById('location_mismatch').value = "Yes";
        document.getElementById('device_change').value = "No";
        document.getElementById('tenure_days').value = 45;
      }} else if (t === 'hold') {{
        document.getElementById('txn_id').value = "TXN_HOLD_03";
        document.getElementById('amount').value = 1280;
        document.getElementById('merchant_avg').value = 200;
        document.getElementById('hour_of_day').value = 3;
        document.getElementById('velocity').value = 8;
        document.getElementById('location_mismatch').value = "Yes";
        document.getElementById('device_change').value = "Yes";
        document.getElementById('tenure_days').value = 15;
      }}
      evaluatePython();
    }}
  </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))
