import subprocess
import sys
from pathlib import Path
import json

import pandas as pd
import mlflow

from src.monitoring.drift import detect_drift


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telco_customer_churn.csv"
)

PRODUCTION_LOG = (
    PROJECT_ROOT
    / "logs"
    / "predictions.csv"
)


def check_drift():

    if not REFERENCE_DATA.exists():
        print("Training dataset not found.")
        return False

    if not PRODUCTION_LOG.exists():
        print("Production prediction log not found.")
        return False

    reference_df = pd.read_csv(REFERENCE_DATA)
    current_df = pd.read_csv(PRODUCTION_LOG)

    if len(current_df) < 10:
        print(
            f"Not enough production data. "
            f"Found {len(current_df)}, need at least 10."
        )
        return False

    drift_results = detect_drift(
        reference_df,
        current_df
    )

    print("\nDRIFT RESULTS")
    print("=" * 40)
    print(drift_results.to_string(index=False))

    high_drift = (
        drift_results["status"] == "HIGH"
    ).any()

    return high_drift


def get_current_model_f1():

    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    client = mlflow.MlflowClient()

    registered_model = client.get_registered_model(
        "ChurnGuardModel"
    )

    latest_versions = registered_model.latest_versions

    if not latest_versions:
        print("No registered model version found.")
        return None

    latest_version = max(
        latest_versions,
        key=lambda v: int(v.version)
    )

    run = client.get_run(
        latest_version.run_id
    )

    current_f1 = run.data.metrics.get("f1")

    if current_f1 is None:
        print(
            "F1 metric not found for "
            f"model version {latest_version.version}."
        )
        return None

    print(
        f"Current registered model: "
        f"Version {latest_version.version}"
    )

    return current_f1


def retrain(current_f1):

    print("\nHigh drift detected.")
    print("Starting model retraining...\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.training.train"
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:

        print(result.stderr)
        print("Retraining failed.")

        return

    result_path = (
        PROJECT_ROOT
        / "models"
        / "latest_training_result.json"
    )

    if not result_path.exists():

        print(
            "Training result file not found."
        )

        return

    with open(result_path, "r") as f:
        new_result = json.load(f)

    new_f1 = new_result["f1"]


    print(f"\nCurrent production F1: {current_f1:.4f}")
    print(f"New model F1   : {new_f1:.4f}")

    if new_f1 > current_f1:
        print("\nNew model performs better.")
        print("Model promotion approved.")

        mlflow.set_tracking_uri(
            f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
        )

        model_uri = (
            f"runs:/{new_result['run_id']}/model"
        )

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name="ChurnGuardModel"
        )

        print("\nNew model registered successfully!")
        print(
            "Model version:",
            registered_model.version
        )

    else:

        print("\nNew model does not improve performance.")
        print("Model promotion rejected.")

def main():

    drift_detected = check_drift()

    if not drift_detected:

        print("\nNo HIGH drift detected.")
        print("Retraining not required.")

        return

    current_f1 = get_current_model_f1()

    if current_f1 is None:

        print("\nCould not determine current model F1.")
        print("Retraining stopped.")

        return

    print(
        f"\nCurrent production F1: "
        f"{current_f1:.4f}"
    )

    retrain(current_f1)

if __name__ == "__main__":
    main()