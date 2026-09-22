from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use a minimal mock for FeatureEngineer to test API contract
# independently without loading the full production feature pipeline

feature_engineer_module = types.ModuleType("src.feature_engineer")


class FakeFeatureEngineer:
    MODEL_FEATURES = ["loanamount"]

    def build_features(self, current_loans, previous_loans, demographics):
        return current_loans[
            ["systemloanid", "customerid", "loanamount"]
        ].copy()


feature_engineer_module.FeatureEngineer = FakeFeatureEngineer
sys.modules.setdefault("src.feature_engineer", feature_engineer_module)

import numpy as np
from fastapi.testclient import TestClient

from api.main import app, get_risk_service
from src.risk_prediction_service import RiskPredictionService


class FakeModelManager:
    def model_name(self):
        return "FakeClassifier"

    def model_version(self):
        return "test"

    def risk_event(self):
        return "Bad"

    def decision_threshold(self):
        return 0.23

    def bad_class_index(self):
        # Deliberately points to the second probability column.
        return 1

    def predict_proba(self, X):
        scores = X["loanamount"].to_numpy(dtype=float)
        bad = np.clip(scores / 100_000.0, 0, 1)
        return np.column_stack([1 - bad, bad])

    def info(self):
        return {
            "model_name": self.model_name(),
            "model_version": self.model_version(),
            "risk_event": self.risk_event(),
            "decision_threshold": self.decision_threshold(),
            "artifact": "fake.joblib",
            "metadata_artifact": "fake.json",
        }


service = RiskPredictionService(
    feature_engineer=FakeFeatureEngineer(),
    model_manager=FakeModelManager(),
)

app.dependency_overrides[get_risk_service] = lambda: service
client = TestClient(app)


def valid_current(loan_id: int = 1, loanamount: float = 20_000) -> dict:
    return {
        "systemloanid": loan_id,
        "customerid": 123,
        "creationdate": "2017-07-01 10:00:00",
        "loanamount": loanamount,
        "totaldue": loanamount * 1.2,
        "termdays": 30,
        "loannumber": 2,
        "referredby": None,
    }


def test_single_prediction_uses_bad_class_probability():
    response = client.post(
        "/predict",
        json={
            "current_loan": valid_current(1, 30_000),
            "previous_loans": [],
            "demographics": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "Bad"
    assert payload["bad_risk_score"] == 0.3
    assert payload["decision_threshold"] == 0.23
    assert payload["model_version"] == "test"


def test_batch_prediction_returns_one_result_per_application():
    response = client.post(
        "/predict/batch",
        json={
            "current_loans": [
                valid_current(1, 10_000),
                valid_current(2, 40_000),
            ],
            "previous_loans": [],
            "demographics": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert [row["systemloanid"] for row in payload["predictions"]] == [1, 2]


def test_duplicate_systemloanid_is_rejected():
    response = client.post(
        "/predict/batch",
        json={
            "current_loans": [
                valid_current(1, 10_000),
                valid_current(1, 20_000),
            ],
            "previous_loans": [],
            "demographics": [],
        },
    )
    assert response.status_code == 422


def test_target_label_is_rejected():
    row = valid_current(1, 10_000)
    row["good_bad_flag"] = "Good"
    response = client.post(
        "/predict",
        json={
            "current_loan": row,
            "previous_loans": [],
            "demographics": [],
        },
    )
    assert response.status_code == 422
