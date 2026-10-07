import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn 
import json 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.preprocessing import (
    prepare_data,
    build_preprocessor
)


def train():

    # Load data
    df = pd.read_csv(
        "data/raw/telco_customer_churn.csv"
    )

    # Prepare X and y
    X, y = prepare_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    }

    # ChurnGuard operating threshold
    threshold = 0.40

    best_f1 = -1
    best_model_name = None
    best_run_id = None

    # MLflow tracking database
    project_root = Path(__file__).resolve().parents[2]

    mlflow.set_tracking_uri(
        f"sqlite:///{project_root / 'mlflow.db'}"
    )

    # MLflow experiment
    mlflow.set_experiment("ChurnGuard")

    for model_name, classifier in models.items():

        with mlflow.start_run(run_name=model_name):

            print(f"\n{'=' * 40}")
            print(f"Model: {model_name}")
            print(f"{'=' * 40}")

            # Build preprocessing pipeline
            preprocessor = build_preprocessor(X_train)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ])

            # Train
            pipeline.fit(X_train, y_train)

            # Predictions
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            # Print metrics
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print(f"ROC-AUC  : {roc_auc:.4f}")

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log threshold
            mlflow.log_param(
                "threshold",
                threshold
            )

            # Log model
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format="cloudpickle"
            )

            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_run_id = mlflow.active_run().info.run_id

    print("\n" + "=" * 40)
    print("BEST MODEL")
    print("=" * 40)

    print("Model :", best_model_name)
    print("F1    :", round(best_f1, 4))
    print("Run ID:", best_run_id) 

    
    result = {
        "model_name": best_model_name,
        "f1": best_f1,
        "run_id": best_run_id
    }

    result_path = project_root / "models" / "latest_training_result.json"

    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)

    return result           
                

if __name__ == "__main__":
    train()