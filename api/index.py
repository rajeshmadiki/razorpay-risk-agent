import os
import sys
import json
import pandas as pd
from http.server import BaseHTTPRequestHandler

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.model import load_data, train_fraud_model, FEATURE_COLS
    from src.agent import FraudAgent
except Exception as e:
    pass

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

            data_path = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
            df = load_data(data_path)
            model, metrics, feat_importances, (X_train, X_test, y_train, y_test) = train_fraud_model(df)
            
            agent = FraudAgent(
                model=model,
                feature_importances=feat_importances,
                feature_means=X_train.mean(),
                feature_stds=X_train.std()
            )

            prob, decision, top_feature = agent.evaluate_transaction(row)

            response_payload = {
                "fraud_probability": round(prob, 4),
                "decision": decision,
                "top_contributing_feature": top_feature,
                "amount_deviation_ratio": dev_ratio,
                "is_night": is_night
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
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "Razorpay Risk Agent Serverless API Online"}).encode('utf-8'))
