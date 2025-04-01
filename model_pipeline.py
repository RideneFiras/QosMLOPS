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


# Configure Elasticsearch logging
ELASTICSEARCH_URL = "http://localhost:9200"  # "http://elasticsearch:9200"# Change  "http://localhost:9200"  to "http://elasticsearch:9200" when containerized
INDEX_NAME = "mlflow-logs"  # Elasticsearch index name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_to_elasticsearch(data):
    """Send log data to Elasticsearch."""
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


# Prepare Data Function
def prepare_data():
    """Loads and prepares the dataset, then saves processed data."""
    print("Loading data...")
    train_df = pd.read_csv("Train.csv")
    test_df = pd.read_csv("Test.csv")

    # Split inputs and targets
    train_inputs = train_df.drop(columns=["target"])
    train_targets = train_df["target"]
    test_inputs = test_df.copy()

    # Select features
    dropped_columns = ["device", "id"]
    train_inputs.drop(columns=dropped_columns, inplace=True)
    test_inputs.drop(columns=dropped_columns, inplace=True)

    # Transform categorical features
    categorical_features = ["area"]
    oe = OrdinalEncoder()
    train_inputs[categorical_features] = oe.fit_transform(
        train_inputs[categorical_features]
    )
    test_inputs[categorical_features] = oe.transform(test_inputs[categorical_features])

    # Missing value imputation
    train_inputs.fillna(0, inplace=True)
    test_inputs.fillna(0, inplace=True)

    # Split training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        train_inputs, train_targets, test_size=0.2, random_state=0
    )

    # Save processed data
    joblib.dump((X_train, X_test, y_train, y_test, test_inputs), "processed_data.pkl")
    print("Data preparation complete. Saved as processed_data.pkl.")


# Train Model Function
def train_model():
    """Loads processed data, trains the model, and logs it to MLflow & Elasticsearch."""
    print("Loading processed data...")
    X_train, X_test, y_train, y_test, _ = joblib.load("processed_data.pkl")

    with mlflow.start_run():
        print("Training model...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        # Log model parameters to MLflow
        mlflow.log_param("n_estimators", rf.n_estimators)
        mlflow.log_param("max_depth", rf.max_depth)

        # Save model
        joblib.dump(rf, "best_rf_model.pkl")
        print("Model training complete. Saved as best_rf_model.pkl.")

        # Log model to MLflow
        mlflow.sklearn.log_model(rf, "random_forest_model")

        # Log training info to Elasticsearch
        log_data = {
            "event": "training_completed",
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "status": "success",
        }
        send_to_elasticsearch(log_data)

    print("Model logged to MLflow & Elasticsearch.")


# Evaluate Model Function
def evaluate_model():
    """Loads the trained model and evaluates its performance."""
    print("Loading model and data for evaluation...")
    X_train, X_test, y_train, y_test, _ = joblib.load("processed_data.pkl")
    rf = joblib.load("best_rf_model.pkl")

    print("Validating model...")
    val_predictions = rf.predict(X_test)
    rmse = mean_squared_error(y_test, val_predictions) ** 0.5
    mae = mean_absolute_error(y_test, val_predictions)  # Mean Absolute Error
    r2 = r2_score(y_test, val_predictions)  # Fix indentation

    print(f"Root Mean Squared Error = {rmse / 1e6:.3f} Mbit/s")
    print(f"🔹 MAE: {mae:.2f} Mbps")
    print(f"🔹 R² Score: {r2:.4f}")

    timestamp = int(time.time() * 1000)

    # Log RMSE to MLflow
    with mlflow.start_run():
        mlflow.log_metric("RMSE", rmse)

    # Send RMSE to Elasticsearch
    log_data = {
        "event": "evaluation_completed",
        "rmse_mbit_s": rmse / 1e6,
        "evaluation_timestamp": datetime.utcfromtimestamp(timestamp / 1000).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    send_to_elasticsearch(log_data)

    print("Evaluation metrics logged to MLflow & Elasticsearch.")


# Save Predictions Function
def save_predictions():
    """Loads the model, makes predictions on the test set, and saves them."""
    print("Loading model for predictions...")
    _, _, _, _, test_inputs = joblib.load("processed_data.pkl")
    rf = joblib.load("best_rf_model.pkl")

    print("Generating predictions...")
    test_predictions = rf.predict(test_inputs)

    # Save predictions
    predictions_df = pd.DataFrame({"id": test_inputs.index, "target": test_predictions})
    predictions_df.to_csv("BenchmarkSubmission.csv", index=False)
    print("Predictions saved as BenchmarkSubmission.csv.")


# Load Model Function
def load_model():
    """Loads the trained model."""
    print("Loading trained model...")
    return joblib.load("best_rf_model.pkl")
