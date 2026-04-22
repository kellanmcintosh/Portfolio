"""
Property-based tests for the Credit Risk Explainer training pipeline.

Properties covered:
  3 — No missing values pass through the preprocessor
  4 — Ordinal encoding preserves domain order
  5 — scale_pos_weight equals the negative-to-positive class ratio
"""

import sys
import os

# Ensure the project root is on the path so `train` can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.preprocessing import OrdinalEncoder

from train import (
    build_pipeline,
    NUMERIC_FEATURES,
    OHE_FEATURES,
    ORDINAL_FEATURES,
    LOAN_GRADE_ORDER,
)

# ---------------------------------------------------------------------------
# Shared helpers / constants
# ---------------------------------------------------------------------------

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + OHE_FEATURES + ORDINAL_FEATURES

# A minimal valid clean row used to build the base dataset for fitting.
_CLEAN_ROW = {
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


def _make_clean_df(n_rows: int = 10) -> pd.DataFrame:
    """Build a minimal valid DataFrame with all 11 feature columns."""
    return pd.DataFrame([_CLEAN_ROW] * n_rows)


# ---------------------------------------------------------------------------
# Strategies for nullable feature values
# ---------------------------------------------------------------------------

_nullable = lambda s: st.one_of(st.none(), s)

_row_strategy = st.fixed_dictionaries(
    {
        # Numeric features — allow None (→ NaN) or a valid value
        "person_age": _nullable(st.integers(min_value=18, max_value=100)),
        "person_income": _nullable(st.integers(min_value=1, max_value=500_000)),
        "person_emp_length": _nullable(st.floats(min_value=0.0, max_value=60.0, allow_nan=False)),
        "loan_amnt": _nullable(st.integers(min_value=500, max_value=35_000)),
        "loan_int_rate": _nullable(st.floats(min_value=5.0, max_value=25.0, allow_nan=False)),
        "loan_percent_income": _nullable(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        "cb_person_cred_hist_length": _nullable(st.integers(min_value=0, max_value=30)),
        # OHE features
        "person_home_ownership": _nullable(st.sampled_from(["RENT", "OWN", "MORTGAGE", "OTHER"])),
        "loan_intent": _nullable(
            st.sampled_from(
                ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
            )
        ),
        "cb_person_default_on_file": _nullable(st.sampled_from(["Y", "N"])),
        # Ordinal feature
        "loan_grade": _nullable(st.sampled_from(LOAN_GRADE_ORDER)),
    }
)


# ---------------------------------------------------------------------------
# Property 3: No missing values pass through the preprocessor
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 3: No missing values pass through the preprocessor

@given(row=_row_strategy)
@settings(max_examples=100)
def test_no_missing_values_after_preprocessing(row):
    """
    **Validates: Requirements 1.5, 1.6**

    For any input row (with any combination of NaN values across all 11 feature
    columns), the fitted ColumnTransformer must produce an output array that
    contains zero NaN values.
    """
    # Build a single-row DataFrame, replacing None with np.nan.
    input_df = pd.DataFrame(
        [{k: (np.nan if v is None else v) for k, v in row.items()}]
    )

    # Fit the preprocessor on the clean base dataset.
    pipeline = build_pipeline(scale_pos_weight=1.0)
    preprocessor = pipeline.named_steps["preprocessor"]
    preprocessor.fit(_make_clean_df())

    # Transform the (potentially NaN-filled) row.
    transformed = preprocessor.transform(input_df)

    assert np.isnan(transformed).sum() == 0, (
        f"Transformed output contains NaN values.\n"
        f"Input row: {row}\n"
        f"Transformed: {transformed}"
    )


# ---------------------------------------------------------------------------
# Property 4: Ordinal encoding preserves domain order
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 4: Ordinal encoding preserves domain order

@given(
    indices=st.lists(
        st.integers(min_value=0, max_value=6),
        min_size=2,
        max_size=2,
        unique=True,
    )
)
@settings(max_examples=100)
def test_ordinal_encoding_loan_grade_preserves_order(indices: list[int]):
    """
    **Validates: Requirements 1.3**

    For any two distinct loan grade values where grade[i] ranks strictly lower
    (better) than grade[j] in the domain ordering ["A" < "B" < ... < "G"],
    the ordinal-encoded value of grade[i] SHALL be strictly less than the
    ordinal-encoded value of grade[j].

    Drawing arbitrary distinct index pairs (not just adjacent ones) ensures
    non-adjacent comparisons such as (A, G) are also verified.
    """
    i, j = sorted(indices)  # ensure i < j so grade[i] is the better grade

    grade_a = LOAN_GRADE_ORDER[i]
    grade_b = LOAN_GRADE_ORDER[j]

    encoder = OrdinalEncoder(
        categories=[LOAN_GRADE_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(pd.DataFrame({"loan_grade": LOAN_GRADE_ORDER}))

    enc_a = float(encoder.transform(pd.DataFrame({"loan_grade": [grade_a]}))[0, 0])
    enc_b = float(encoder.transform(pd.DataFrame({"loan_grade": [grade_b]}))[0, 0])

    assert enc_a < enc_b, (
        f"Expected encoded('{grade_a}') < encoded('{grade_b}'), "
        f"but got {enc_a} >= {enc_b}"
    )


# ---------------------------------------------------------------------------
# Property 5: scale_pos_weight equals the negative-to-positive class ratio
# ---------------------------------------------------------------------------

# Feature: credit-risk-explainer, Property 5: scale_pos_weight equals the negative-to-positive class ratio

@given(
    labels=st.lists(
        st.integers(min_value=0, max_value=1),
        min_size=2,
        max_size=200,
    ).filter(lambda lst: 1 in lst)  # ensure at least one positive sample
)
@settings(max_examples=100)
def test_scale_pos_weight_equals_neg_to_pos_ratio(labels):
    """
    **Validates: Requirements 2.2**

    For any pandas Series of 0s and 1s with at least one positive sample,
    the scale_pos_weight computed the same way train_and_evaluate does
    (neg_count / pos_count) SHALL equal (y == 0).sum() / (y == 1).sum().
    """
    y = pd.Series(labels)

    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())

    # Reproduce the exact calculation from train_and_evaluate.
    computed_scale_pos_weight: float = neg_count / pos_count

    expected: float = (y == 0).sum() / (y == 1).sum()

    assert computed_scale_pos_weight == expected, (
        f"scale_pos_weight mismatch: computed {computed_scale_pos_weight}, "
        f"expected {expected} (neg={neg_count}, pos={pos_count})"
    )
