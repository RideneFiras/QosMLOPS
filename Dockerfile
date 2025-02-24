# Use an official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 

# Expose FastAPI port
EXPOSE 8000

# Run data preparation before training
CMD python -c "from model_pipeline import prepare_data; prepare_data()" && \
    python -c "from model_pipeline import train_model; train_model()" && \
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload