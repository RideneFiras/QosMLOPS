import argparse
import model_pipeline
import mlflow
import mlflow.sklearn
import os


# ✅ Set MLflow tracking URI to Elasticsearch inside Docker
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://elasticsearch:9200")
mlflow.set_tracking_uri(mlflow_uri)


def main():
    """Main script to execute different parts of the ML pipeline."""
    parser = argparse.ArgumentParser(
        description="Run different parts of the ML pipeline."
    )

    parser.add_argument("--prepare", action="store_true", help="Prepare data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Generate predictions")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full pipeline (prepare → train → evaluate → save)",
    )

    args = parser.parse_args()

    if args.all:
        print("Running full pipeline...")
        model_pipeline.prepare_data()
        model_pipeline.train_model()
        model_pipeline.evaluate_model()
        model_pipeline.save_predictions()
        print("Full pipeline execution completed!")

    if args.prepare:
        model_pipeline.prepare_data()

    if args.train:
        model_pipeline.train_model()

    if args.evaluate:
        model_pipeline.evaluate_model()

    if args.predict:
        model_pipeline.save_predictions()


if __name__ == "__main__":
    main()
