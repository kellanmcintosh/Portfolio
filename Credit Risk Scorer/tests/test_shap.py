"""
Tests for SHAP integration in the Credit Risk Explainer API.

Covers:
  Property 7 — SHAP values sum to prediction deviation from base value
  Property 8 — SHAP feature names match pipeline output features

Note on Property 7:
  XGBoost's TreeExplainer computes SHAP values in log-odds space (raw margin
  output), not probability space.  The additivity identity that holds is:

      sigmoid(sum(shap_values) + base_value) ≈ risk_score

  where sigmoid(x) = 1 / (1 + exp(-x)).  The equivalent check used here is:

      abs(sigmoid(shap_sum + base_value) - risk_score) < 1e-4

  This correctly validates Requirements 4.1 and 4.2 (SHAP additivity) while
  accounting for the log-odds output space of the XGBoost TreeExplainer.
"""

import math
import os
import sys

# Ensure the project root is on the path so `api` can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the API at the real pipeline artifact before importing the app.
os.environ.setdefault("MODEL_PATH", "model/pipeline.joblib")

import joblib
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from api.main import app

# ---------------------------------------------------------------------------
# TestClient — module-scoped fixture so the lifespan runs exactly once.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

# ---------------------------------------------------------------------------
# Hypothesis strategies for valid LoanFeatures payloads
# ---------------------------------------------------------------------------

_valid_payload_strategy = st.fixed_dictionaries(
    {
        "person_age": st.integers(min_value=18, max_value=100),
        "person_income": st.integers(min_value=1, max_value=1_000_000),
        "person_home_ownership": st.sampled_from(["RENT", "OWN", "MORTGAGE", "OTHER"]),
        "person_emp_length": st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
        ),
        "loan_intent": st.sampled_from(
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        ),
        "loan_grade": st.sampled_from(["A", "B", "C", "D", "E", "F", "G"]),
        "loan_amnt": st.integers(min_value=1, max_value=100_000),
        "loan_int_rate": st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        ),
        "loan_percent_income": st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        "cb_person_default_on_file": st.sampled_from(["Y", "N"]),
        "cb_person_cred_hist_length": st.integers(min_value=0, max_value=50),
    }
)

# ---------------------------------------------------------------------------
# Task 6.3 — Property 7: SHAP values sum to prediction deviation from base value
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 7: SHAP values sum to prediction deviation from base value


@given(payload=_valid_payload_strategy)
@settings(max_examples=50)
def test_shap_values_sum_to_prediction_deviation(client, payload):
    """
    **Validates: Requirements 4.1, 4.2**

    XGBoost's TreeExplainer returns SHAP values in log-odds space.  The
    additivity identity is:

        sigmoid(sum(shap_values) + base_value) ≈ risk_score

    This verifies that the SHAP explanation is consistent with the model's
    predicted probability, satisfying the "SHAP values sum to prediction
    deviation from base value" property in the correct output space.
    """
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. Body: {response.text}"
    )
    body = response.json()
    risk_score = body["risk_score"]
    base_value = body["base_value"]
    shap_values = body["shap_values"]

    shap_sum = sum(sv["value"] for sv in shap_values)
    # SHAP additivity in log-odds space: sigmoid(shap_sum + base_value) ≈ risk_score
    reconstructed = 1.0 / (1.0 + math.exp(-(shap_sum + base_value)))
    assert abs(reconstructed - risk_score) < 1e-4, (
        f"sigmoid(shap_sum + base_value) = {reconstructed:.6f} does not match "
        f"risk_score = {risk_score:.6f} "
        f"(diff={abs(reconstructed - risk_score):.2e})"
    )


# ---------------------------------------------------------------------------
# Task 6.4 — Property 8: SHAP feature names match pipeline output features
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 8: SHAP feature names match pipeline output features

_VALID_PAYLOAD = {
    "person_age": 30,
    "person_income": 50000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5.0,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 10.5,
    "loan_percent_income": 0.2,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 3,
}


def test_shap_feature_names_match_pipeline_output(client):
    """
    **Validates: Requirements 4.5**

    The set of feature names in the SHAP explanation SHALL equal the set of
    feature names produced by pipeline.named_steps["preprocessor"].get_feature_names_out().
    """
    loaded_pipeline = joblib.load(os.environ.get("MODEL_PATH", "model/pipeline.joblib"))
    expected_features = set(
        loaded_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )

    response = client.post("/predict", json=_VALID_PAYLOAD)
    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. Body: {response.text}"
    )
    body = response.json()
    actual_features = {sv["feature"] for sv in body["shap_values"]}

    assert actual_features == expected_features, (
        f"SHAP feature names do not match pipeline output features.\n"
        f"Missing from SHAP: {expected_features - actual_features}\n"
        f"Extra in SHAP: {actual_features - expected_features}"
    )
