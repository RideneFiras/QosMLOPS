# Define Python executable
PYTHON = python3

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


# Run the full pipeline (prepare → train → evaluate → save)
all: prepare train evaluate 

# Start Jupyter Notebook
notebook:
	@echo "Launching Jupyter Notebook..."
	@jupyter notebook

fastapi:
	uvicorn app:app --reload
	
    

mlflow:
	@echo "Launching MLflow"
	@mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &
	@sleep 2 && open http://127.0.0.1:5000

# Build Docker Compose Services (includes FastAPI)
docker-build:
	@echo "Building services defined in docker-compose.yml..."
	@docker compose build



# Push to Docker Hub
docker-push:
	@read -p " Enter tag (e.g. v1.0, dev): " tag; \
	echo " Building image with tag: $$tag"; \
	docker build -t firasrid/firas-ridene-4data-mlops:$$tag .; \
	echo " Pushing $$tag and latest to Docker Hub..."; \
	docker push firasrid/firas-ridene-4data-mlops:$$tag; \
	


# Clean up cached files
clean:
	@echo "Cleaning up cache and old files..."
	rm -rf __pycache__ .pytest_cache *.pkl *.log


# Run Black (formatting) via pre-commita
format:
	@echo "🖌️ Formatting code with Black..."
	@pre-commit run black --all-files

# Run Flake8 (linting) via pre-commit
lint:
	@echo "🔍 Running Flake8..."
	@pre-commit run flake8 --all-files

# Run both (Black + Flake8)
check:
	@echo "🔍 Running Linting & Formatting..."
	@pre-commit run --all-files
	

# Start only the PostgreSQL service
db:
	@echo "🟢 Starting PostgreSQL container (Docker Compose)..."
	docker compose up db

# Stop and remove all containers from docker-compose
docker-down:
	@echo "🛑 Shutting down all Docker Compose services..."
	docker compose down

services-up:
	docker compose up -d elasticsearch kibana db mlflow


start:
	make services-up
	uvicorn app:app --reload


# 🚨 Monitor system resources (CPU/RAM)
monitor-alerts:
	@echo "📢 Starting system resource monitor..."
	@python3 ciservices/notify_monitor.py

ci:
	@echo "🚀 Running CI pipeline..."
	@{ \
		make monitor-alerts > monitor.log 2>&1 & \
		MON_PID=$$!; \
		make check; \
		make all; \
		kill $$MON_PID; \
	}

	
live_check:
	./ciservices/ci_watch.sh

security-audit:
	@echo "🔐 Running security audit..."
	@bandit -r . -x qos,venv,__pycache__

htop:
	@htop