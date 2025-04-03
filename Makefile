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
	@echo "Launching database in background..."
	docker compose up -d db
	@echo "Starting FastAPI..."
	@open http://127.0.0.1:8000/
	uvicorn app:app --reload
	
    

mlflow:
	@echo "Launching MLflow"
	@mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &
	@sleep 2 && open http://127.0.0.1:5000

# Build Docker Compose Services (includes FastAPI)
docker-build:
	@echo "Building services defined in docker-compose.yml..."
	@docker-compose build

# Run Docker Compose Stack
docker-run:
	@echo "Running full Docker stack with Docker Compose..."
	@docker-compose up -d

# Push to Docker Hub
docker-push:
	@echo "Pushing image to Docker Hub..."
	@docker push firasrid/firas-ridene-4data-mlops:latest

# Clean up cached files
clean:
	@echo "Cleaning up cache and old files..."
	rm -rf __pycache__ .pytest_cache *.pkl *.log


lint:
	@echo "🔍 Running Flake8..."
	@flake8 --ignore=E501 .

# ✅ Run Black (Formatting)
format:
	@echo "🖌️ Formatting code with Black..."
	@black .

# ✅ Run both Flake8 & Black
check:
	@echo "🔍 Running Linting & Formatting..."
	@make format
	@make lint

# Start only the PostgreSQL service
db:
	@echo "🟢 Starting PostgreSQL container (Docker Compose)..."
	docker compose up db

# Stop and remove all containers from docker-compose
docker-down:
	@echo "🛑 Shutting down all Docker Compose services..."
	docker compose down
