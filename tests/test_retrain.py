from unittest.mock import patch, mock_open
import json
import pandas as pd

from src.training import retrain

def test_retrain_rejects_model_without_improvement(tmp_path):
    result_data = {
        "f1": 0.60,
        "run_id": "test_run_123"
    }

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    result_file = models_dir / "latest_training_result.json"
    result_file.write_text(json.dumps(result_data))

    with patch.object(retrain, "PROJECT_ROOT", tmp_path):
        with patch.object(retrain.subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Training completed"
            mock_run.return_value.stderr = ""

            with patch.object(retrain, "mlflow") as mock_mlflow:
                retrain.retrain(current_f1=0.65)

                mock_mlflow.register_model.assert_not_called()

def test_retrain_promotes_better_model(tmp_path):
    result_data = {
        "f1": 0.75,
        "run_id": "test_run_456"
    }

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    result_file = models_dir / "latest_training_result.json"
    result_file.write_text(json.dumps(result_data))
    with patch.object(retrain, "PROJECT_ROOT", tmp_path):
        with patch.object(retrain.subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Training completed"
            mock_run.return_value.stderr = ""

            with patch.object(retrain, "mlflow") as mock_mlflow:
                mock_mlflow.register_model.return_value.version = "3"

                retrain.retrain(current_f1=0.65)

                mock_mlflow.register_model.assert_called_once_with(
                    model_uri="runs:/test_run_456/model",
                    name="ChurnGuardModel"
                )                


def test_check_drift_missing_reference_data(tmp_path):
    with patch.object(retrain, "REFERENCE_DATA", tmp_path / "missing.csv"):
        with patch.object(retrain, "PRODUCTION_LOG", tmp_path / "predictions.csv"):
            assert retrain.check_drift() is False


def test_check_drift_missing_production_log(tmp_path):
    reference_file = tmp_path / "reference.csv"
    reference_file.write_text("tenure\n12\n24\n")

    with patch.object(retrain, "REFERENCE_DATA", reference_file):
        with patch.object(retrain, "PRODUCTION_LOG", tmp_path / "missing.csv"):
            assert retrain.check_drift() is False


def test_check_drift_insufficient_production_records(tmp_path):
    reference_file = tmp_path / "reference.csv"
    reference_file.write_text("tenure\n12\n24\n")

    production_file = tmp_path / "predictions.csv"
    production_file.write_text("tenure\n12\n24\n")

    with patch.object(retrain, "REFERENCE_DATA", reference_file):
        with patch.object(retrain, "PRODUCTION_LOG", production_file):
            assert retrain.check_drift() is False


def test_get_current_model_f1_no_versions():
    with patch.object(retrain.mlflow, "MlflowClient") as mock_client:
        mock_client.return_value.get_registered_model.return_value.latest_versions = []

        result = retrain.get_current_model_f1()

        assert result is None


def test_get_current_model_f1_missing_metric():
    mock_version = type(
        "Version",
        (),
        {"version": "1", "run_id": "test_run"}
    )()

    mock_run = type(
        "Run",
        (),
        {"data": type("Data", (), {"metrics": {}})()}
    )()

    with patch.object(retrain.mlflow, "MlflowClient") as mock_client:
        mock_client.return_value.get_registered_model.return_value.latest_versions = [
            mock_version
        ]
        mock_client.return_value.get_run.return_value = mock_run

        result = retrain.get_current_model_f1()

        assert result is None


def test_get_current_model_f1_success():
    mock_version = type(
        "Version",
        (),
        {"version": "2", "run_id": "test_run"}
    )()

    mock_run = type(
        "Run",
        (),
        {"data": type("Data", (), {"metrics": {"f1": 0.75}})()}
    )()

    with patch.object(retrain.mlflow, "MlflowClient") as mock_client:
        mock_client.return_value.get_registered_model.return_value.latest_versions = [
            mock_version
        ]
        mock_client.return_value.get_run.return_value = mock_run

        result = retrain.get_current_model_f1()

        assert result == 0.75


def test_retrain_training_process_fails(tmp_path):
    with patch.object(retrain, "PROJECT_ROOT", tmp_path):
        with patch.object(retrain.subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "Training failed"

            retrain.retrain(current_f1=0.65)

            assert not (tmp_path / "models" / "latest_training_result.json").exists()


def test_retrain_result_file_missing(tmp_path):
    with patch.object(retrain, "PROJECT_ROOT", tmp_path):
        with patch.object(retrain.subprocess, "run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "Training completed"
            mock_run.return_value.stderr = ""

            retrain.retrain(current_f1=0.65)

            assert not (tmp_path / "models" / "latest_training_result.json").exists()


def test_main_stops_when_no_drift():
    with patch.object(retrain, "check_drift", return_value=False):
        with patch.object(retrain, "get_current_model_f1") as mock_f1:
            with patch.object(retrain, "retrain") as mock_retrain:
                retrain.main()

                mock_f1.assert_not_called()
                mock_retrain.assert_not_called()


def test_main_stops_when_current_f1_missing():
    with patch.object(retrain, "check_drift", return_value=True):
        with patch.object(retrain, "get_current_model_f1", return_value=None):
            with patch.object(retrain, "retrain") as mock_retrain:
                retrain.main()

                mock_retrain.assert_not_called()


def test_main_triggers_retraining_when_drift_detected():
    with patch.object(retrain, "check_drift", return_value=True):
        with patch.object(retrain, "get_current_model_f1", return_value=0.65):
            with patch.object(retrain, "retrain") as mock_retrain:
                retrain.main()

                mock_retrain.assert_called_once_with(0.65)


import pandas as pd


def test_check_drift_detects_high_drift(tmp_path):
    reference_file = tmp_path / "reference.csv"
    reference_file.write_text("tenure\n12\n24\n36\n")

    production_file = tmp_path / "predictions.csv"
    production_file.write_text(
        "tenure\n12\n24\n36\n48\n60\n72\n84\n96\n108\n120\n"
    )

    drift_results = pd.DataFrame({
        "feature": ["tenure", "MonthlyCharges"],
        "psi": [0.30, 0.05],
        "status": ["HIGH", "LOW"]
    })

    with patch.object(retrain, "REFERENCE_DATA", reference_file):
        with patch.object(retrain, "PRODUCTION_LOG", production_file):
            with patch.object(
                retrain, "detect_drift", return_value=drift_results
            ) as mock_detect:
                result = retrain.check_drift()

    mock_detect.assert_called_once()
    assert result == True


def test_check_drift_no_high_drift(tmp_path):
    reference_file = tmp_path / "reference.csv"
    reference_file.write_text("tenure\n12\n24\n36\n")

    production_file = tmp_path / "predictions.csv"
    production_file.write_text(
        "tenure\n12\n24\n36\n48\n60\n72\n84\n96\n108\n120\n"
    )

    drift_results = pd.DataFrame({
        "feature": ["tenure", "MonthlyCharges"],
        "psi": [0.05, 0.15],
        "status": ["LOW", "MEDIUM"]
    })

    with patch.object(retrain, "REFERENCE_DATA", reference_file):
        with patch.object(retrain, "PRODUCTION_LOG", production_file):
            with patch.object(
                retrain, "detect_drift", return_value=drift_results
            ):
                result = retrain.check_drift()

    assert result == False



def test_main_prints_f1_and_triggers_retraining(capsys):
    with patch.object(retrain, "check_drift", return_value=True):
        with patch.object(retrain, "get_current_model_f1", return_value=0.75):
            with patch.object(retrain, "retrain") as mock_retrain:
                retrain.main()

                mock_retrain.assert_called_once_with(0.75)

    captured = capsys.readouterr()
    assert "Current production F1: 0.7500" in captured.out
