import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.MlflowClient()

versions = client.search_model_versions(
    "name='ChurnGuardModel'"
)

for version in versions:
    print("=" * 50)
    print("Name:", version.name)
    print("Version:", version.version)
    print("Run ID:", version.run_id)
    print("Source:", version.source)
    