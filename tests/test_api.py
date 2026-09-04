import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app

class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertEqual(data["model_name"], "RandomForestClassifier")
        self.assertEqual(data["mode"], "DEFENSE-ONLY")

    def test_02_risk_score_clear_scenario(self):
        payload = {
            "transaction_id": "TXN_TEST_CLEAR",
            "amount": 25.0,
            "merchant_avg_amount": 50.0,
            "hour_of_day": 14,
            "velocity_last_hour": 1,
            "location_mismatch": "No",
            "device_change": "No",
            "customer_tenure_days": 600
        }
        response = self.client.post("/api/v1/risk/score", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["decision"], "CLEAR")
        self.assertLess(data["fraud_probability"], 0.40)

    def test_03_risk_score_hold_scenario(self):
        payload = {
            "transaction_id": "TXN_TEST_HOLD",
            "amount": 1280.0,
            "merchant_avg_amount": 200.0,
            "hour_of_day": 3,
            "velocity_last_hour": 8,
            "location_mismatch": "Yes",
            "device_change": "Yes",
            "customer_tenure_days": 15
        }
        response = self.client.post("/api/v1/risk/score", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["decision"], "HOLD")
        self.assertGreaterEqual(data["fraud_probability"], 0.75)

    def test_04_batch_endpoint(self):
        response = self.client.post("/api/v1/risk/batch")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_transactions"], 600)
        self.assertIn("clear_count", data)
        self.assertIn("hold_count", data)

    def test_05_audit_endpoint(self):
        response = self.client.get("/api/v1/audit?decision=HOLD")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_06_evaluation_endpoint(self):
        response = self.client.get("/api/v1/evaluation")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("precision", data)
        self.assertIn("recall", data)
        self.assertIn("f1_score", data)

    def test_07_audit_verify_endpoint(self):
        # Run batch endpoint to ensure audit log is populated with cryptographic hashes
        self.client.post("/api/v1/risk/batch")
        response = self.client.get("/api/v1/audit/verify")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_valid"])
        self.assertIsNone(data["tampered_index"])
        self.assertGreater(data["total_records"], 0)

    def test_08_cost_analysis_endpoint(self):
        response = self.client.get("/api/v1/evaluation/cost")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("false_positive_count", data)
        self.assertIn("illustrative_total_fp_cost", data)
        self.assertIn("false_negative_value_exposure", data)
        self.assertEqual(data["model_version"], "fraud-rf-v1")

if __name__ == "__main__":
    unittest.main()


