"""
Credit Risk Explainer — Streamlit UI

Application states
------------------
ZERO_STATE  — initial load; shows a placeholder message
LOADING     — spinner while the API call is in progress
SUCCESS     — renders prediction label + risk score + SHAP waterfall plot
ERROR       — user-friendly message; no crash (handled inside call_predict_api)
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests
import shap
import streamlit as st


# ---------------------------------------------------------------------------
# Task 8.1 — build_sidebar_form
# ---------------------------------------------------------------------------

def build_sidebar_form() -> dict[str, Any]:
    """Render labeled sidebar input controls for all 11 loan features.

    Returns
    -------
    dict[str, Any]
        Keys match the ``LoanFeatures`` field names expected by the API.
        Optional fields (``person_emp_length``, ``loan_int_rate``) are
        returned as ``None`` when the user leaves them at 0.0.
    """
    st.sidebar.header("Loan Application Details")

    person_age: int = st.sidebar.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
    )

    person_income: int = st.sidebar.number_input(
        "Annual Income ($)",
        min_value=1,
        value=50_000,
        step=1_000,
    )

    person_home_ownership: str = st.sidebar.selectbox(
        "Home Ownership",
        options=["RENT", "OWN", "MORTGAGE", "OTHER"],
    )

    _emp_length_unknown: bool = st.sidebar.checkbox(
        "Employment length unknown",
        value=False,
    )
    if _emp_length_unknown:
        person_emp_length: float | None = None
    else:
        person_emp_length = st.sidebar.number_input(
            "Employment Length (years)",
            min_value=0.0,
            value=5.0,
            step=0.5,
        )

    loan_intent: str = st.sidebar.selectbox(
        "Loan Intent",
        options=[
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION",
        ],
    )

    loan_grade: str = st.sidebar.selectbox(
        "Loan Grade",
        options=["A", "B", "C", "D", "E", "F", "G"],
    )

    loan_amnt: int = st.sidebar.number_input(
        "Loan Amount ($)",
        min_value=1,
        value=10_000,
        step=500,
    )

    _int_rate_unknown: bool = st.sidebar.checkbox(
        "Interest rate unknown",
        value=False,
    )
    if _int_rate_unknown:
        loan_int_rate: float | None = None
    else:
        loan_int_rate = st.sidebar.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            value=10.5,
            step=0.1,
        )

    loan_percent_income: float = st.sidebar.slider(
        "Loan as % of Income",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
    )

    cb_person_default_on_file: str = st.sidebar.selectbox(
        "Previous Default on File",
        options=["N", "Y"],
    )

    cb_person_cred_hist_length: int = st.sidebar.number_input(
        "Credit History Length (years)",
        min_value=0,
        value=3,
        step=1,
    )

    return {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
    }


# ---------------------------------------------------------------------------
# Task 8.2 — call_predict_api
# ---------------------------------------------------------------------------

def call_predict_api(payload: dict, base_url: str) -> dict | None:
    """POST the loan features to the prediction API and return the response.

    Parameters
    ----------
    payload:
        Dict matching the ``LoanFeatures`` schema.
    base_url:
        Base URL of the prediction service, e.g. ``http://api:8000``.

    Returns
    -------
    dict | None
        Parsed JSON response on HTTP 200, or ``None`` on any error (an
        appropriate ``st.error`` message is displayed before returning).
    """
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=payload,
            timeout=10,
        )
    except requests.ConnectionError:
        st.error(
            "Could not connect to the prediction service. "
            "Please ensure the API is running."
        )
        return None
    except Exception:  # noqa: BLE001
        st.error("An unexpected error occurred.")
        return None

    if response.status_code != 200:
        st.error(
            f"The prediction service returned an error "
            f"(HTTP {response.status_code})."
        )
        return None

    return response.json()


# ---------------------------------------------------------------------------
# Task 8.3 — render_prediction
# ---------------------------------------------------------------------------

def render_prediction(response: dict) -> None:
    """Display the prediction label (colour-coded) and risk score percentage.

    Parameters
    ----------
    response:
        Parsed ``PredictionResponse`` dict from the API.
    """
    prediction: str = response["prediction"]
    risk_score: float = response["risk_score"]

    color = "red" if prediction == "High Risk" else "green"

    st.markdown(
        f"<h2 style='color: {color};'>{prediction}</h2>",
        unsafe_allow_html=True,
    )
    st.write(f"Risk Score: {risk_score * 100:.1f}%")


# ---------------------------------------------------------------------------
# Task 8.4 — render_shap_plot
# ---------------------------------------------------------------------------

def render_shap_plot(shap_values: list[dict], base_value: float) -> None:
    """Reconstruct a SHAP Explanation and render a Waterfall Plot.

    Parameters
    ----------
    shap_values:
        List of ``{"feature": str, "value": float}`` dicts from the API.
    base_value:
        The model's expected value (base value) returned by the API.
    """
    st.subheader("Feature Contributions (SHAP)")

    # Sort by absolute contribution magnitude (largest first) so the waterfall
    # plot leads with the most influential features.
    sorted_shap = sorted(shap_values, key=lambda sv: abs(sv["value"]), reverse=True)

    explanation = shap.Explanation(
        values=np.array([sv["value"] for sv in sorted_shap]),
        base_values=base_value,
        feature_names=[sv["feature"] for sv in sorted_shap],
    )

    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 8.5 — main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Wire all application states into the Streamlit entry point."""
    st.set_page_config(page_title="Credit Risk Explainer", layout="wide")
    st.title("Credit Risk Explainer")

    api_base_url: str = os.environ.get("API_BASE_URL", "http://api:8000")

    # Collect form inputs from the sidebar.
    payload = build_sidebar_form()

    # "Predict" button lives in the sidebar, below the form controls.
    predict_clicked = st.sidebar.button("Predict")

    # Initialise session state on first load.
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "has_predicted" not in st.session_state:
        st.session_state.has_predicted = False

    if predict_clicked:
        with st.spinner("Analysing loan application..."):
            result = call_predict_api(payload, api_base_url)

        if result is not None:
            st.session_state.prediction_result = result
            st.session_state.has_predicted = True
        else:
            # Error already displayed inside call_predict_api.
            st.session_state.has_predicted = False
            st.session_state.prediction_result = None

    # Render the appropriate state.
    if not st.session_state.has_predicted:
        st.info(
            "Fill in the loan details in the sidebar and click Predict."
        )
    else:
        result = st.session_state.prediction_result
        render_prediction(result)
        render_shap_plot(result["shap_values"], result["base_value"])


if __name__ == "__main__":
    main()
