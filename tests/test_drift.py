
import pandas as pd

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
