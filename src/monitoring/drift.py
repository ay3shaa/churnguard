import pandas as pd
import numpy as np


NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10
) -> float:

    reference = pd.to_numeric(
        reference,
        errors="coerce"
    ).dropna()

    current = pd.to_numeric(
        current,
        errors="coerce"
    ).dropna()

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    reference_pct = np.clip(
        reference_pct,
        0.0001,
        None
    )

    current_pct = np.clip(
        current_pct,
        0.0001,
        None
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame
):

    results = []

    for feature in NUMERICAL_FEATURES:

        psi = calculate_psi(
            reference_df[feature],
            current_df[feature]
        )

        if psi < 0.10:
            status = "LOW"

        elif psi < 0.25:
            status = "MEDIUM"

        else:
            status = "HIGH"

        results.append({
            "feature": feature,
            "psi": round(psi, 4),
            "status": status
        })

    return pd.DataFrame(results)