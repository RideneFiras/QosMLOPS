# 📦 Updated model_pipeline.py with new data preparation and XGBoost training

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import logging
import requests
import time
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Detect environment
IS_DOCKER = os.getenv("IS_DOCKER", "false").lower() == "true"

# Configure dynamic Elasticsearch URL based on environment
ELASTICSEARCH_URL = (
    "http://elasticsearch:9200" if IS_DOCKER else "http://localhost:9200"
)
INDEX_NAME = "mlflow-logs"

# Configure dynamic MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5000" if IS_DOCKER else "http://localhost:5000")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_to_elasticsearch(data):
    try:
        response = requests.post(
            f"{ELASTICSEARCH_URL}/{INDEX_NAME}/_doc",
            json=data,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code not in [200, 201]:
            logger.error(f"Failed to send log to Elasticsearch: {response.text}")
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")


def prepare_data():
    print("Loading and cleaning data...")
    df = pd.read_csv("Dataset/Train.csv")

    # --- Your new cleaning logic ---

    absence_activite_scell = (
        df[
            [
                "SCell_Cell_Identity",
                "SCell_RSRP_max",
                "SCell_RSRQ_max",
                "SCell_RSSI_max",
                "SCell_SNR_max",
                "SCell_Downlink_Num_RBs",
                "SCell_Downlink_Average_MCS",
                "SCell_Downlink_bandwidth_MHz",
            ]
        ]
        .isnull()
        .any(axis=1)
    )
    df["SCell_Active"] = np.where(absence_activite_scell, 0, 1)

    mask_active = df["SCell_Active"] == 1
    for col in [
        "SCell_RSRP_max",
        "SCell_RSRQ_max",
        "SCell_RSSI_max",
        "SCell_SNR_max",
        "SCell_Downlink_Num_RBs",
        "SCell_Downlink_Average_MCS",
        "SCell_Downlink_bandwidth_MHz",
    ]:
        if col in df.columns:
            df.loc[mask_active, col] = df[col].fillna(df[col].mean())

    mask_inactive = df["SCell_Active"] == 0
    columns_to_zero = [
        "SCell_Cell_Identity",
        "SCell_RSRP_max",
        "SCell_RSRQ_max",
        "SCell_RSSI_max",
        "SCell_SNR_max",
        "SCell_freq_MHz",
        "SCell_Downlink_Num_RBs",
        "SCell_Downlink_Average_MCS",
        "SCell_Downlink_bandwidth_MHz",
    ]
    for col in columns_to_zero:
        if col in df.columns:
            df.loc[mask_inactive & df[col].isnull(), col] = 0

    cols_to_drop = [
        "visibility",
        "windSpeed",
        "SCell_freq_MHz",
        "PCell_freq_MHz",
        "uvIndex",
        "COG",
        "precipIntensity",
        "Pressure",
        "id",
        "PCell_Cell_Identity",
        "SCell_Cell_Identity",
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["weekday"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    if "PCell_SNR_max" in df.columns and "PCell_RSRP_max" in df.columns:
        df["RSRP_SNR_ratio"] = df["PCell_SNR_max"] / (df["PCell_RSRP_max"].abs() + 1e-3)
    if "PCell_SNR_max" in df.columns and "PCell_RSRQ_max" in df.columns:
        df["RSRQ_SNR_ratio"] = df["PCell_SNR_max"] / (df["PCell_RSRQ_max"].abs() + 1e-3)
    if (
        "PCell_Downlink_Num_RBs" in df.columns
        and "PCell_Downlink_Average_MCS" in df.columns
    ):
        df["estimated_utilization"] = (
            df["PCell_Downlink_Num_RBs"] * df["PCell_Downlink_Average_MCS"]
        )

    df = df.select_dtypes(include=[np.number])
    df = df.dropna()

    target_column = "target"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    joblib.dump((X_train, X_test, y_train, y_test), "Models/processedx_data.pkl")
    print("Data preparation complete. Saved as processedx_data.pkl.")


def train_model():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = joblib.load("Models/processedx_data.pkl")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        subsample=0.8,
        n_estimators=500,
        min_child_weight=5,
        max_depth=6,
        learning_rate=0.05,
        gamma=0.5,
        colsample_bytree=0.6,
        verbosity=0,
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    joblib.dump(model, "Models/best_xgb_model.pkl")
    print("✅ Model training complete. Saved as best_xgb_model.pkl.")

    try:
        with mlflow.start_run():
            mlflow.xgboost.log_model(model, "xgboost_model")
            mlflow.log_params({
                "n_estimators": model.get_params()["n_estimators"],
                "max_depth": model.get_params()["max_depth"],
                "learning_rate": model.get_params()["learning_rate"]
            })
            print("✅ Model logged to MLflow.")
    except Exception as e:
        print(f"⚠️ Skipped MLflow logging: {e}")

    log_data = {
        "event": "training_completed",
        "model": "xgboost",
        "status": "success",
    }

    try:
        send_to_elasticsearch(log_data)
        print("✅ Log sent to Elasticsearch.")
    except Exception as e:
        print(f"⚠️ Skipped Elasticsearch logging: {e}")



def evaluate_model():
    print("Loading model and data for evaluation...")
    X_train, X_test, y_train, y_test = joblib.load("Models/processedx_data.pkl")
    model = joblib.load("Models/best_xgb_model.pkl")

    print("Validating model...")
    val_predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, val_predictions))
    mae = mean_absolute_error(y_test, val_predictions)
    r2 = r2_score(y_test, val_predictions)

    rmse_mbps = rmse / 1e6
    mae_mbps = mae / 1e6

    print(f"Root Mean Squared Error = {rmse_mbps:.4f} Mbps")
    print(f"MAE = {mae_mbps:.4f} Mbps")
    print(f"R² Score = {r2:.4f}")

    with mlflow.start_run():
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

    timestamp = int(time.time() * 1000)

    log_data = {
        "event": "evaluation_completed",
        "rmse_mbps": rmse_mbps,
        "mae_mbps": mae_mbps,
        "r2_score": r2,
        "evaluation_timestamp": datetime.utcfromtimestamp(timestamp / 1000).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    send_to_elasticsearch(log_data)

    print("Evaluation metrics logged to MLflow & Elasticsearch.")


def load_model():
    print("Loading trained model...")
    return joblib.load("Models/xgboost.pkl")
