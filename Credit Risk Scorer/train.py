"""
Training pipeline for the Credit Risk Explainer.

Dataset: data/credit_risk_dataset.csv
Target:  loan_status  (0 = no default, 1 = default)

Functions
---------
load_data          -- Load the credit risk dataset from a CSV file.
build_pipeline     -- Construct the preprocessing + XGBoost Pipeline.
train_and_evaluate -- Fit, evaluate, and persist the Pipeline.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: list[str] = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

OHE_FEATURES: list[str] = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

ORDINAL_FEATURES: list[str] = ["loan_grade"]

LOAN_GRADE_ORDER: list[str] = ["A", "B", "C", "D", "E", "F", "G"]

TARGET: str = "loan_status"


# ---------------------------------------------------------------------------
# Task 2.1 — Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load the credit risk dataset from *path*.

    Parameters
    ----------
    path:
        Relative or absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with the original column names preserved.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please ensure 'data/credit_risk_dataset.csv' is present in the "
            "project root before running the training script."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Task 2.2 — Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Construct the full preprocessing + classifier Pipeline.

    The ColumnTransformer applies three branches:

    * ``num``     — median imputation → standard scaling on all numeric cols
                    (handles the 895 nulls in person_emp_length and 3116 in
                    loan_int_rate).
    * ``cat_ohe`` — most-frequent imputation → one-hot encoding on nominal
                    categorical cols.
    * ``cat_ord`` — most-frequent imputation → ordinal encoding on loan_grade
                    (A=0 best … G=6 worst).

    Parameters
    ----------
    scale_pos_weight:
        Ratio of negative-class samples to positive-class samples, passed
        directly to :class:`xgboost.XGBClassifier`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted Pipeline ready for ``fit`` / ``predict``.
    """
    num_branch = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_ohe_branch = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    cat_ord_branch = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[LOAN_GRADE_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num",     num_branch,     NUMERIC_FEATURES),
        ("cat_ohe", cat_ohe_branch, OHE_FEATURES),
        ("cat_ord", cat_ord_branch, ORDINAL_FEATURES),
    ])

    classifier = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# Task 2.3 — Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, output_path: str) -> None:
    """Fit the Pipeline on *df*, log ROC-AUC, and persist the artifact.

    Parameters
    ----------
    df:
        Raw DataFrame as returned by :func:`load_data`.
    output_path:
        Destination path for the serialised Pipeline (e.g.
        ``"model/pipeline.joblib"``).  The parent directory is created
        automatically if it does not exist.

    Raises
    ------
    ValueError
        If the dataset contains zero positive-class (default) samples, making
        it impossible to compute a meaningful ``scale_pos_weight``.
    """
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    pos_count: int = int((y == 1).sum())
    neg_count: int = int((y == 0).sum())

    if pos_count == 0:
        raise ValueError(
            "The dataset contains no positive-class (default) samples. "
            "Cannot compute scale_pos_weight (division by zero)."
        )

    scale_pos_weight: float = neg_count / pos_count

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(scale_pos_weight=scale_pos_weight)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_prob)
    print(f"Validation ROC-AUC: {roc_auc:.4f}")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Pipeline saved to '{output_path}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data("data/credit_risk_dataset.csv")
    train_and_evaluate(df, "model/pipeline.joblib")
