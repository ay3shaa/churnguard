
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.monitoring.drift import calculate_psi, detect_drift


def test_calculate_psi_same_distribution():
    reference = pd.Series([10, 20, 30, 40, 50, 60])
    current = reference.copy()

    psi = calculate_psi(reference, current)

    assert psi >= 0
    assert psi < 0.10


def test_calculate_psi_empty_data():
    reference = pd.Series([], dtype=float)
    current = pd.Series([10, 20, 30])

    assert calculate_psi(reference, current) == 0.0


def test_detect_drift_returns_all_features():
    reference = pd.DataFrame({
        "tenure": [1, 5, 10, 20, 30, 40, 50, 60],
        "MonthlyCharges": [20, 30, 40, 50, 60, 70, 80, 90],
        "TotalCharges": [20, 150, 400, 1000, 1800, 2800, 4000, 5500],
    })

    current = reference.copy()

    results = detect_drift(reference, current)

    assert len(results) == 3
    assert set(results["feature"]) == {
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    }
    assert set(results["status"]).issubset({"LOW", "MEDIUM", "HIGH"})

def test_calculate_psi_constant_reference():
    reference = pd.Series([5, 5, 5, 5, 5])
    current = pd.Series([1, 2, 3, 4, 5])

    result = calculate_psi(reference, current)

    assert result == 0.0

def test_detect_drift_medium_status():
    import pandas as pd
    from src.monitoring.drift import detect_drift

    reference_df = pd.DataFrame({
        "tenure": [1, 2, 3],
        "MonthlyCharges": [20, 30, 40],
        "TotalCharges": [100, 200, 300]
    })

    current_df = reference_df.copy()

    with patch(
        "src.monitoring.drift.calculate_psi",
        side_effect=[0.05, 0.15, 0.30]
    ):
        result = detect_drift(reference_df, current_df)

    assert result["status"].tolist() == ["LOW", "MEDIUM", "HIGH"]