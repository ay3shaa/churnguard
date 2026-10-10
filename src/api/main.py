import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

import time
from datetime import datetime,timezone

import shap
import numpy as np

from src.monitoring.drift import detect_drift




app = FastAPI(
    title="ChurnGuard API",
    description="Customer churn prediction API",
    version="1.0.0"
)


import mlflow.sklearn
from pathlib import Path

MODEL_PATH = Path("models/churnguard")

model = mlflow.sklearn.load_model(str(MODEL_PATH))

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()

explainer = shap.LinearExplainer(
    classifier,
    np.zeros((1, len(feature_names)))
)

LOG_FILE = Path("logs/predictions.csv")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("data/raw/telco_customer_churn.csv")

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str


@app.get("/")
def root():
    return {
        "message": "ChurnGuard API is running"
    }

def get_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    return "LOW"


@app.post("/predict")
def predict(customer: Customer):

    start_time = time.perf_counter()

    data = pd.DataFrame([customer.model_dump()])

    probability = model.predict_proba(data)[0][1]

    transformed_data = preprocessor.transform(data)

    shap_values = explainer.shap_values(transformed_data)

    shap_values = np.asarray(shap_values).flatten()

    top_indices = np.argsort(np.abs(shap_values))[::-1][:5]

    top_features = []

    for index in top_indices:
        feature_name = feature_names[index]

        feature_name = (
            feature_name
            .replace("numerical__", "")
            .replace("categorical__", "")
            .replace("_", " ")
        )

        top_features.append({
            "feature": feature_name,
            "impact": round(float(shap_values[index]), 4)
        })

    threshold = 0.40

    prediction = int(probability >= threshold)

    risk_level = get_risk_level(probability)

    response_time = (time.perf_counter() - start_time) * 1000

    log_entry = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "tenure": customer.tenure,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": float(customer.TotalCharges),

        "churn_probability": float(probability),
        "churn_prediction": prediction,
        "risk_level": risk_level,
        "response_time_ms": round(response_time, 2),

        "factor_1": top_features[0]["feature"],
        "impact_1": top_features[0]["impact"],

        "factor_2": top_features[1]["feature"],
        "impact_2": top_features[1]["impact"],

        "factor_3": top_features[2]["feature"],
        "impact_3": top_features[2]["impact"],

        "factor_4": top_features[3]["feature"],
        "impact_4": top_features[3]["impact"],

        "factor_5": top_features[4]["feature"],
        "impact_5": top_features[4]["impact"]
    }])

    log_entry.to_csv(
        LOG_FILE,
        mode="a",
        header=not LOG_FILE.exists(),
        index=False
    )

    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction,
        "risk_level": risk_level,
        "response_time_ms": round(response_time, 2),
        "top_factors": top_features
    }

    
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "ChurnGuardModel"
    }


@app.get("/drift")
def drift():

    if not DATA_PATH.exists():
        return {
            "error": "Training dataset not found"
        }

    reference_df = pd.read_csv(DATA_PATH)

    current_df = pd.read_csv(LOG_FILE)

    if len(current_df) < 10:
        return {
            "message": "Not enough production data for drift detection",
            "minimum_required": 10,
            "current_records": len(current_df)
        }

    drift_results = detect_drift(
        reference_df,
        current_df
    )

    return {
        "drift_detected": bool(
            (drift_results["status"] == "HIGH").any()
        ),
        "results": drift_results.to_dict(
            orient="records"
        )
    }