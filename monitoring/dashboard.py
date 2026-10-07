import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="ChurnGuard Monitoring",
    layout="wide"
)

st.title("ChurnGuard Monitoring Dashboard")
st.caption("Production prediction monitoring")

LOG_FILE = Path("logs/predictions.csv")

if not LOG_FILE.exists():
    st.warning("No prediction logs found yet.")
    st.stop()

df = pd.read_csv(LOG_FILE)

if df.empty:
    st.warning("Prediction log is empty.")
    st.stop()

# -------------------------
# Convert timestamp
# -------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])

# -------------------------
# Metrics
# -------------------------

total_predictions = len(df)

predicted_churn = int(
    (df["churn_prediction"] == 1).sum()
)

churn_rate = (
    predicted_churn / total_predictions * 100
)

avg_latency = df["response_time_ms"].mean()

# -------------------------
# KPI Cards
# -------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Predictions",
    total_predictions
)

col2.metric(
    "Predicted Churn",
    predicted_churn
)

col3.metric(
    "Predicted Churn Rate",
    f"{churn_rate:.1f}%"
)

col4.metric(
    "Avg API Latency",
    f"{avg_latency:.2f} ms"
)

st.divider()

# -------------------------
# Risk Distribution
# -------------------------

st.subheader("Risk Distribution")

risk_counts = (
    df["risk_level"]
    .value_counts()
    .reindex(["LOW", "MEDIUM", "HIGH"])
    .fillna(0)
)

st.bar_chart(risk_counts)

# -------------------------
# Prediction Distribution
# -------------------------

st.subheader("Prediction Distribution")

prediction_counts = (
    df["churn_prediction"]
    .value_counts()
    .rename(index={
        0: "No Churn",
        1: "Churn"
    })
)

st.bar_chart(prediction_counts)

# -------------------------
# API Latency
# -------------------------

st.subheader("API Response Time")

latency_df = df.set_index("timestamp")[
    ["response_time_ms"]
]

st.line_chart(latency_df)

# -------------------------
# Recent Predictions
# -------------------------

st.subheader("Recent Predictions")

st.dataframe(
    df.sort_values(
        "timestamp",
        ascending=False
    ).head(20),
    use_container_width=True
)

# -------------------------
# Prediction Explainability
# -------------------------

st.divider()

st.subheader("Prediction Explainability")

latest = df.sort_values(
    "timestamp",
    ascending=False
).iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latest Churn Probability",
        f"{latest['churn_probability'] * 100:.2f}%"
    )

with col2:
    st.metric(
        "Latest Risk Level",
        latest["risk_level"]
    )

# Check whether SHAP columns exist
required_columns = []

for i in range(1, 6):
    required_columns.extend([
        f"factor_{i}",
        f"impact_{i}"
    ])

has_shap_data = all(
    column in df.columns
    for column in required_columns
)

if has_shap_data:

    st.write("### Top Factors")

    factors = []

    for i in range(1, 6):

        feature = latest[f"factor_{i}"]
        impact = latest[f"impact_{i}"]

        factors.append({
            "Feature": feature,
            "SHAP Impact": impact,
            "Effect": (
                "Increases churn"
                if impact > 0
                else "Reduces churn"
            )
        })

    factor_df = pd.DataFrame(factors)

    st.dataframe(
        factor_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "SHAP explanations are not available for the existing "
        "prediction logs. Generate a new prediction to view "
        "explainability results."
    )