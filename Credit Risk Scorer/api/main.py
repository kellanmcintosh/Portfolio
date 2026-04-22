"""
Credit Risk Explainer — Prediction API

Endpoints
---------
GET  /health   — liveness check; returns {"status": "ok"}
POST /predict  — accepts LoanFeatures JSON, returns PredictionResponse JSON
"""

import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.environ.get("MODEL_PATH", "model/pipeline.joblib")

# Module-level pipeline and SHAP explainer references; populated during startup.
pipeline = None  # type: ignore[assignment]
explainer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Task 4.1 — Pydantic schemas
# ---------------------------------------------------------------------------

class LoanFeatures(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: int = Field(..., ge=1)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: Optional[float] = Field(None, ge=0.0)
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: int = Field(..., ge=1)
    loan_int_rate: Optional[float] = Field(None, ge=0.0)
    loan_percent_income: float = Field(..., ge=0.0, le=1.0)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0)


class ShapFeatureValue(BaseModel):
    feature: str
    value: float


class PredictionResponse(BaseModel):
    risk_score: float
    prediction: str
    shap_values: list[ShapFeatureValue]
    base_value: float


# ---------------------------------------------------------------------------
# Task 4.2 — Startup lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML pipeline and initialise the SHAP explainer on startup."""
    global pipeline, explainer
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info("Pipeline loaded from '%s'", MODEL_PATH)
    except FileNotFoundError:
        logger.error(
            "Pipeline artifact not found at '%s'. "
            "Run 'python train.py' to generate it before starting the API.",
            MODEL_PATH,
        )
        sys.exit(1)
    # Build the TreeExplainer once so it isn't reconstructed on every request.
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    logger.info("SHAP TreeExplainer initialised")
    yield
    # Nothing to clean up for joblib artifacts.


app = FastAPI(title="Credit Risk Explainer API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Task 4.2 — GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Liveness check — returns HTTP 200 only when the pipeline is loaded."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Task 6.1 — compute_shap
# ---------------------------------------------------------------------------

def compute_shap(df: pd.DataFrame) -> tuple[list[ShapFeatureValue], float]:
    """Compute SHAP values for a single-row DataFrame.

    Uses the module-level ``explainer`` cached at startup — TreeExplainer
    construction is expensive and must not be repeated per request.

    Parameters
    ----------
    df:
        A single-row DataFrame with the raw (unpreprocessed) feature columns.

    Returns
    -------
    tuple[list[ShapFeatureValue], float]
        A list of per-feature SHAP values and the base value (expected value).
    """
    preprocessed = pipeline.named_steps["preprocessor"].transform(df)
    shap_vals = explainer.shap_values(preprocessed)
    # For binary XGBoost, shap_values returns a 2D array; take the single row.
    row_shap = shap_vals[0]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    shap_feature_values = [
        ShapFeatureValue(feature=str(name), value=float(val))
        for name, val in zip(feature_names, row_shap)
    ]
    # expected_value may be a scalar or a 1-element array for binary XGBoost.
    expected = explainer.expected_value
    if hasattr(expected, "__len__"):
        base_value = float(expected[0])
    else:
        base_value = float(expected)
    return shap_feature_values, base_value


# ---------------------------------------------------------------------------
# Task 4.3 / 6.2 — POST /predict
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(features: LoanFeatures) -> PredictionResponse:
    """Run inference and return risk score + prediction label + SHAP values.

    HTTP 422 is returned automatically by FastAPI for Pydantic validation
    failures.  Any unexpected exception during inference returns HTTP 500.
    """
    try:
        df = pd.DataFrame([features.model_dump()])
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        risk_score: float = float(pipeline.predict_proba(df)[0][1])
        prediction: str = "High Risk" if risk_score >= 0.5 else "Low Risk"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction.",
        ) from exc

    try:
        shap_values, base_value = compute_shap(df)
    except Exception as exc:  # noqa: BLE001
        logger.error("SHAP computation failed:\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during SHAP computation.",
        ) from exc

    return PredictionResponse(
        risk_score=risk_score,
        prediction=prediction,
        shap_values=shap_values,
        base_value=base_value,
    )
