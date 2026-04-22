"""
Tests for the Credit Risk Explainer prediction API.

Covers:
  Property 1  — Risk score is a valid probability
  Property 2  — Prediction label is consistent with risk score threshold
  Property 6  — Schema validation rejects all out-of-range and invalid-enum inputs
  Property 9  — Prediction is deterministic
  Integration — /health, POST /predict (valid), POST /predict (invalid)
"""

import os
import sys

# Ensure the project root is on the path so `api` can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the API at the real pipeline artifact before importing the app.
os.environ.setdefault("MODEL_PATH", "model/pipeline.joblib")

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings, assume
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
# Task 4.4 — Property 1: Risk score is a valid probability
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 1: Risk score is a valid probability

@given(payload=_valid_payload_strategy)
@settings(max_examples=100)
def test_risk_score_is_valid_probability(client, payload):
    """
    **Validates: Requirements 3.5**

    For any valid LoanFeatures input, the risk_score returned by POST /predict
    SHALL be a float in the closed interval [0.0, 1.0].
    """
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. Body: {response.text}"
    )
    risk_score = response.json()["risk_score"]
    assert 0.0 <= risk_score <= 1.0, (
        f"risk_score {risk_score} is outside [0.0, 1.0] for payload: {payload}"
    )


# ---------------------------------------------------------------------------
# Task 4.5 — Property 2: Prediction label is consistent with threshold
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 2: Prediction label is consistent with risk score threshold

@given(payload=_valid_payload_strategy)
@settings(max_examples=100)
def test_prediction_label_consistent_with_threshold(client, payload):
    """
    **Validates: Requirements 3.6, 3.7**

    For any valid LoanFeatures input:
      - prediction == "High Risk"  iff  risk_score >= 0.5
      - prediction == "Low Risk"   iff  risk_score <  0.5
    """
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. Body: {response.text}"
    )
    body = response.json()
    risk_score = body["risk_score"]
    prediction = body["prediction"]

    if risk_score >= 0.5:
        assert prediction == "High Risk", (
            f"Expected 'High Risk' for risk_score={risk_score}, got '{prediction}'"
        )
    else:
        assert prediction == "Low Risk", (
            f"Expected 'Low Risk' for risk_score={risk_score}, got '{prediction}'"
        )


# ---------------------------------------------------------------------------
# Task 4.6 — Property 6: Schema validation rejects invalid inputs
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 6: Schema validation rejects all out-of-range and invalid-enum inputs

# Base valid payload used as a template for mutation.
_BASE_VALID_PAYLOAD = {
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

# Strategy that generates a single invalid mutation to apply to the base payload.
_invalid_mutation_strategy = st.one_of(
    # person_age out of range
    st.just({"person_age": 17}),
    st.just({"person_age": 101}),
    st.integers(max_value=17).map(lambda v: {"person_age": v}),
    st.integers(min_value=101).map(lambda v: {"person_age": v}),
    # person_income out of range
    st.just({"person_income": 0}),
    st.integers(max_value=0).map(lambda v: {"person_income": v}),
    # loan_amnt out of range
    st.just({"loan_amnt": 0}),
    st.integers(max_value=0).map(lambda v: {"loan_amnt": v}),
    # loan_percent_income out of range
    st.just({"loan_percent_income": -0.01}),
    st.just({"loan_percent_income": 1.01}),
    st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False).map(
        lambda v: {"loan_percent_income": v}
    ),
    st.floats(min_value=1.001, allow_nan=False, allow_infinity=False).map(
        lambda v: {"loan_percent_income": v}
    ),
    # cb_person_cred_hist_length out of range
    st.just({"cb_person_cred_hist_length": -1}),
    st.integers(max_value=-1).map(lambda v: {"cb_person_cred_hist_length": v}),
    # Invalid enum values
    st.just({"person_home_ownership": "CONDO"}),
    st.just({"loan_intent": "GAMBLING"}),
    st.just({"loan_grade": "H"}),
    st.just({"cb_person_default_on_file": "X"}),
    # person_emp_length out of range (negative)
    st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False).map(
        lambda v: {"person_emp_length": v}
    ),
    # loan_int_rate out of range (negative)
    st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False).map(
        lambda v: {"loan_int_rate": v}
    ),
)


@given(mutation=_invalid_mutation_strategy)
@settings(max_examples=100)
def test_schema_validation_rejects_invalid_inputs(client, mutation):
    """
    **Validates: Requirements 3.4, 3.8**

    For any LoanFeatures payload where at least one field violates its
    constraint, POST /predict SHALL return HTTP 422.
    """
    invalid_payload = {**_BASE_VALID_PAYLOAD, **mutation}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422, (
        f"Expected HTTP 422 for invalid payload {invalid_payload}, "
        f"got {response.status_code}. Body: {response.text}"
    )


# ---------------------------------------------------------------------------
# Task 4.7 — Property 9: Prediction is deterministic
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 9: Prediction is deterministic

@given(payload=_valid_payload_strategy)
@settings(max_examples=50)
def test_prediction_is_deterministic(client, payload):
    """
    **Validates: Requirements 3.5, 3.6, 3.7**

    Calling POST /predict twice with identical inputs SHALL return identical
    risk_score and prediction values.
    """
    response1 = client.post("/predict", json=payload)
    response2 = client.post("/predict", json=payload)

    assert response1.status_code == 200, (
        f"First call returned {response1.status_code}. Body: {response1.text}"
    )
    assert response2.status_code == 200, (
        f"Second call returned {response2.status_code}. Body: {response2.text}"
    )

    body1 = response1.json()
    body2 = response2.json()

    assert body1["risk_score"] == body2["risk_score"], (
        f"risk_score differs between calls: {body1['risk_score']} vs {body2['risk_score']}"
    )
    assert body1["prediction"] == body2["prediction"], (
        f"prediction differs between calls: {body1['prediction']} vs {body2['prediction']}"
    )


# ---------------------------------------------------------------------------
# Task 4.8 — Integration tests
# ---------------------------------------------------------------------------

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


def test_health_returns_200_ok(client):
    """GET /health → HTTP 200 with body {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload_returns_200_with_valid_structure(client):
    """POST /predict with a valid payload → HTTP 200 with valid PredictionResponse."""
    response = client.post("/predict", json=_VALID_PAYLOAD)
    assert response.status_code == 200, f"Body: {response.text}"

    body = response.json()

    # Required fields present
    assert "risk_score" in body
    assert "prediction" in body
    assert "shap_values" in body
    assert "base_value" in body

    # Type and range checks
    assert isinstance(body["risk_score"], float)
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["prediction"] in ("High Risk", "Low Risk")
    assert isinstance(body["shap_values"], list)
    assert isinstance(body["base_value"], float)


def test_predict_invalid_payload_returns_422(client):
    """POST /predict with age=17 (below minimum) → HTTP 422."""
    invalid_payload = {**_VALID_PAYLOAD, "person_age": 17}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
