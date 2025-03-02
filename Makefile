# Define Python executable
PYTHON = python3.10

# Data Preparation: Load and process features
prepare:
	@echo "Preparing data..."
	@$(PYTHON) main.py --prepare

# Train the model
train:
	@echo "Training..."
	@$(PYTHON) main.py --train

# Evaluate the model
evaluate:
	@echo "Evaluating saved model..."
	@$(PYTHON) main.py --evaluate

# Generate predictions
predict:
	@echo "Generating predictions..."
	@$(PYTHON) main.py --predict

# Run the full pipeline (prepare → train → evaluate → save)
all: prepare train evaluate predict

# Start Jupyter Notebook
notebook:
	@echo "Launching Jupyter Notebook..."
	@jupyter notebook

fastapi:
	@echo "launching fastapi and webpage"
	@open http://127.0.0.1:8000/ && uvicorn app:app --reload

mlflow:
	@echo "Launching MLflow"
	@mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &
	@sleep 2 && open http://127.0.0.1:5000

# Build Docker Image
docker-build:
	@echo "Building Docker image..."
	@docker build -t firas-ridene-4data-mlops .

# Run Docker Container
docker-run:
	@echo "Running FastAPI in Docker..."
	@docker run -p 8000:8000 firas-ridene-4data-mlops

# Push to Docker Hub
docker-push:
	@echo "Pushing image to Docker Hub..."
	@docker push firasrid/firas-ridene-4data-mlops:latest

# Clean up cached files
clean:
	@echo "Cleaning up cache and old files..."
	rm -rf __pycache__ .pytest_cache *.pkl *.log


