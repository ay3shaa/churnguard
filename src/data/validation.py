import pandas as pd


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the dataset has the expected columns.
    """

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )


def validate_duplicates(df: pd.DataFrame) -> None:
    """
    Validate that there are no duplicate customer IDs.
    """

    duplicate_ids = df["customerID"].duplicated().sum()

    if duplicate_ids > 0:
        raise ValueError(
            f"Found {duplicate_ids} duplicate customer IDs"
        )


def validate_target(df: pd.DataFrame) -> None:
    """
    Validate target column values.
    """

    valid_values = {"Yes", "No"}

    invalid_values = set(df["Churn"].unique()) - valid_values

    if invalid_values:
        raise ValueError(
            f"Invalid Churn values: {invalid_values}"
        )


def validate_data(df: pd.DataFrame) -> None:
    """
    Run all data validation checks.
    """

    validate_schema(df)
    validate_duplicates(df)
    validate_target(df)

    print("Data validation passed.")