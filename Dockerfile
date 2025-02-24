# Use an official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y make git && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 

# Expose FastAPI and MLflow ports
EXPOSE 8000 5000

# Start MLflow tracking server and run FastAPI
CMD mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 & \
    make all && uvicorn app:app --host 0.0.0.0 --port 8000 --reload