import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Prepare Data Function
def prepare_data():
    """Loads and prepares the dataset, then saves processed data."""
    print("Loading data...")
    train_df = pd.read_csv("Train.csv")
    test_df = pd.read_csv("Test.csv")

    # Split inputs and targets
    train_inputs = train_df.drop(columns=['target'])
    train_targets = train_df['target']
    test_inputs = test_df.copy()

    # Select features
    dropped_columns = ['device', 'id']
    train_inputs.drop(columns=dropped_columns, inplace=True)
    test_inputs.drop(columns=dropped_columns, inplace=True)

    # Transform categorical features
    categorical_features = ['area']
    oe = OrdinalEncoder()
    train_inputs[categorical_features] = oe.fit_transform(train_inputs[categorical_features])
    test_inputs[categorical_features] = oe.transform(test_inputs[categorical_features])

    # Missing value imputation
    train_inputs.fillna(0, inplace=True)
    test_inputs.fillna(0, inplace=True)

    # Split training and validation tests
    X_train, X_test, y_train, y_test = train_test_split(train_inputs, train_targets, test_size=0.2, random_state=0)

    # Save processed data
    joblib.dump((X_train, X_test, y_train, y_test, test_inputs), "processed_data.pkl")
    print("Data preparation complete. Saved as processed_data.pkl.")

# Train Model Function
def train_model():
    """Loads processed data, trains the model, and logs it to MLflow."""
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

        # Log the model as an artifact in MLflow
        mlflow.sklearn.log_model(rf, "random_forest_model")

    print("Model logged to MLflow.")

# Evaluate Model Function
def evaluate_model():
    """Loads the trained model and evaluates its performance."""
    print("Loading model and data for evaluation...")
    X_train, X_test, y_train, y_test, _ = joblib.load("processed_data.pkl")
    rf = joblib.load("best_rf_model.pkl")

    print("Validating model...")
    val_predictions = rf.predict(X_test)
    rmse = mean_squared_error(y_test, val_predictions)**0.5
    print(f"Root Mean Squared Error = {rmse / 1e6:.3} Mbit/s")

    # Log RMSE to MLflow
    with mlflow.start_run():
        mlflow.log_metric("RMSE", rmse)

# Save Predictions Function
def save_predictions():
    """Loads the model, makes predictions on the test set, and saves them."""
    print("Loading model for predictions...")
    _, _, _, _, test_inputs = joblib.load("processed_data.pkl")
    rf = joblib.load("best_rf_model.pkl")

    print("Generating predictions...")
    test_predictions = rf.predict(test_inputs)

    # Save predictions
    predictions_df = pd.DataFrame({'id': test_inputs.index, 'target': test_predictions})
    predictions_df.to_csv("BenchmarkSubmission.csv", index=False)
    print("Predictions saved as BenchmarkSubmission.csv.")

# Load Model Function
def load_model():
    """Loads the trained model."""
    print("Loading trained model...")
    return joblib.load("best_rf_model.pkl")
