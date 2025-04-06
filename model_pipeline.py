import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import logging
import requests
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
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
    print("Loading data...")
    train_df = pd.read_csv("Dataset/Train.csv")
    test_df = pd.read_csv("Dataset/Test.csv")

    train_inputs = train_df.drop(columns=["target"])
    train_targets = train_df["target"]
    test_inputs = test_df.copy()

    dropped_columns = ["device", "id"]
    train_inputs.drop(columns=dropped_columns, inplace=True)
    test_inputs.drop(columns=dropped_columns, inplace=True)

    categorical_features = ["area"]
    oe = OrdinalEncoder()
    train_inputs[categorical_features] = oe.fit_transform(
        train_inputs[categorical_features]
    )
    test_inputs[categorical_features] = oe.transform(test_inputs[categorical_features])

    train_inputs.fillna(0, inplace=True)
    test_inputs.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        train_inputs, train_targets, test_size=0.2, random_state=0
    )

    joblib.dump(
        (X_train, X_test, y_train, y_test, test_inputs), "Models/processed_data.pkl"
    )
    print("Data preparation complete. Saved as processed_data.pkl.")


def train_model():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test, _ = joblib.load("Models/processed_data.pkl")

    with mlflow.start_run():
        print("Training model...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        mlflow.log_param("n_estimators", rf.n_estimators)
        mlflow.log_param("max_depth", rf.max_depth)

        joblib.dump(rf, "Models/best_rf_model.pkl")
        print("Model training complete. Saved as best_rf_model.pkl.")

        mlflow.sklearn.log_model(rf, "random_forest_model")

        log_data = {
            "event": "training_completed",
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "status": "success",
        }
        send_to_elasticsearch(log_data)

    print("Model logged to MLflow & Elasticsearch.")


def evaluate_model():
    print("Loading model and data for evaluation...")
    X_train, X_test, y_train, y_test, _ = joblib.load("Models/processed_data.pkl")
    rf = joblib.load("Models/best_rf_model.pkl")

    print("Validating model...")
    val_predictions = rf.predict(X_test)
    rmse = mean_squared_error(y_test, val_predictions) ** 0.5
    mae = mean_absolute_error(y_test, val_predictions)
    r2 = r2_score(y_test, val_predictions)
    print(f"Root Mean Squared Error = {rmse / 1e6:.3f} Mbit/s")
    print(f"MAE: {mae:.2f} Mbps")
    print(f"R² Score: {r2:.4f}")

    timestamp = int(time.time() * 1000)

    with mlflow.start_run():
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

    log_data = {
        "event": "evaluation_completed",
        "rmse_mbit_s": rmse / 1e6,
        "mae_mbit_s": mae / 1e6,
        "r2_score": r2,
        "evaluation_timestamp": datetime.utcfromtimestamp(timestamp / 1000).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    send_to_elasticsearch(log_data)

    print("Evaluation metrics logged to MLflow & Elasticsearch.")


def save_predictions():
    print("Loading model for predictions...")
    _, _, _, _, test_inputs = joblib.load("Models/processed_data.pkl")
    rf = joblib.load("Models/best_rf_model.pkl")

    print("Generating predictions...")
    test_predictions = rf.predict(test_inputs)

    predictions_df = pd.DataFrame({"id": test_inputs.index, "target": test_predictions})
    predictions_df.to_csv("BenchmarkSubmission.csv", index=False)
    print("Predictions saved as BenchmarkSubmission.csv.")


def load_model():
    print("Loading trained model...")
    return joblib.load("Models/best_rf_model.pkl")
