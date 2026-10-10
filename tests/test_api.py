
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.main import get_risk_level

from unittest.mock import patch

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "ChurnGuard API is running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "ChurnGuardModel"


def test_predict_endpoint():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": "786.0"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in [0, 1]
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(result["top_factors"]) == 5



def test_predict_rejects_missing_required_field():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5
        # TotalCharges is intentionally missing
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_invalid_field_type():
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": "not-a-number",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": "786.0"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_risk_level_low():
    assert get_risk_level(0.39) == "LOW"


def test_risk_level_medium():
    assert get_risk_level(0.40) == "MEDIUM"


def test_risk_level_high():
    assert get_risk_level(0.70) == "HIGH"


def test_drift_missing_training_data():
    with patch("src.api.main.DATA_PATH", new_callable=lambda: __import__("pathlib").Path("nonexistent_training_file.csv")):
        response = client.get("/drift")

    assert response.status_code == 200
    assert response.json() == {
        "error": "Training dataset not found"
    }

def test_drift_insufficient_production_data():
    with patch("src.api.main.DATA_PATH") as mock_data_path:
        mock_data_path.exists.return_value = True

        with patch("src.api.main.pd.read_csv") as mock_read_csv:
            import pandas as pd

            mock_read_csv.side_effect = [
                pd.DataFrame({"tenure": [1, 2, 3]}),
                pd.DataFrame({"tenure": [1, 2, 3]})
            ]

            response = client.get("/drift")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Not enough production data for drift detection"
    )
    assert response.json()["minimum_required"] == 10
    assert response.json()["current_records"] == 3    

def test_drift_successful_detection():
    import pandas as pd

    reference_df = pd.DataFrame({"tenure": [1, 2, 3]})
    current_df = pd.DataFrame({"tenure": list(range(10))})

    mock_results = pd.DataFrame({
        "feature": ["tenure"],
        "psi": [0.30],
        "status": ["HIGH"]
    })

    with patch("src.api.main.DATA_PATH") as mock_data_path:
        mock_data_path.exists.return_value = True

        with patch("src.api.main.pd.read_csv",
                   side_effect=[reference_df, current_df]):
            with patch("src.api.main.detect_drift",
                       return_value=mock_results):
                response = client.get("/drift")

    assert response.status_code == 200

    data = response.json()
    assert data["drift_detected"] is True
    assert len(data["results"]) == 1
    assert data["results"][0]["status"] == "HIGH"    