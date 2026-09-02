import os
import unittest
import pandas as pd
import numpy as np

from src.generate_data import generate_synthetic_transactions
from src.model import load_data, train_fraud_model, FEATURE_COLS, TARGET_COL
from src.agent import FraudAgent

class TestFraudAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic_transactions(n_samples=200, random_seed=123)
        cls.model, cls.metrics, cls.feat_imp, (cls.X_train, cls.X_test, cls.y_train, cls.y_test) = train_fraud_model(cls.df, random_state=123)
        cls.feature_means = cls.X_train.mean()
        cls.feature_stds = cls.X_train.std()

    def test_01_data_loading_and_columns(self):
        """Verify generated dataset contains all required interpretable feature and target columns."""
        expected_cols = set(FEATURE_COLS + [TARGET_COL, "transaction_id"])
        self.assertTrue(expected_cols.issubset(set(self.df.columns)))
        self.assertGreater(len(self.df), 0)

    def test_02_model_training_and_metrics(self):
        """Verify model fit completes and evaluation metrics are valid probabilities."""
        self.assertIn("precision", self.metrics)
        self.assertIn("recall", self.metrics)
        self.assertIn("f1", self.metrics)

        for metric_name, val in self.metrics.items():
            self.assertGreaterEqual(val, 0.0, f"{metric_name} should be >= 0.0")
            self.assertLessEqual(val, 1.0, f"{metric_name} should be <= 1.0")

    def test_03_decision_threshold_boundaries(self):
        """Test policy thresholds: >=0.75 HOLD, 0.40-0.75 ESCALATE, <0.40 CLEAR."""
        agent = FraudAgent(
            model=self.model,
            feature_importances=self.feat_imp,
            feature_means=self.feature_means,
            feature_stds=self.feature_stds
        )

        sample_row = self.df.iloc[0].copy()

        # Mock model's predict_proba for controlled test
        class DummyModel:
            def __init__(self, proba):
                self.proba = proba
            def predict_proba(self, X):
                return np.array([[1.0 - self.proba, self.proba]])

        # Test HOLD threshold
        agent_hold = FraudAgent(DummyModel(0.80), self.feat_imp, self.feature_means, self.feature_stds)
        prob, dec, _ = agent_hold.evaluate_transaction(sample_row)
        self.assertEqual(dec, "HOLD")
        self.assertGreaterEqual(prob, 0.75)

        # Test ESCALATE threshold
        agent_escalate = FraudAgent(DummyModel(0.50), self.feat_imp, self.feature_means, self.feature_stds)
        prob, dec, _ = agent_escalate.evaluate_transaction(sample_row)
        self.assertEqual(dec, "ESCALATE")
        self.assertTrue(0.40 <= prob < 0.75)

        # Test CLEAR threshold
        agent_clear = FraudAgent(DummyModel(0.20), self.feat_imp, self.feature_means, self.feature_stds)
        prob, dec, _ = agent_clear.evaluate_transaction(sample_row)
        self.assertEqual(dec, "CLEAR")
        self.assertLess(prob, 0.40)

    def test_04_safety_gate_downgrade(self):
        """Test that when running HOLD rate exceeds 25%, subsequent HOLDs are downgraded to ESCALATE."""
        class HighRiskModel:
            def predict_proba(self, X):
                return np.array([[0.10, 0.90]])  # Always 90% fraud prob

        agent = FraudAgent(HighRiskModel(), self.feat_imp, self.feature_means, self.feature_stds)
        sample_row = self.df.iloc[0]

        decisions = []
        for _ in range(15):
            _, dec, _ = agent.evaluate_transaction(sample_row)
            decisions.append(dec)

        # First 10 items are within warm-up window (HOLD), after which hold rate is 10/10 = 100% > 25%,
        # triggering safety gate downgrade to ESCALATE for subsequent items.
        self.assertTrue(agent.gate_triggered)
        self.assertIn("HOLD", decisions[:10])
        self.assertIn("ESCALATE", decisions[10:])
        self.assertEqual(decisions[-1], "ESCALATE")

if __name__ == "__main__":
    unittest.main()
