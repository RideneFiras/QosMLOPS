# Use an official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 

# Run full pipeline (data processing + training) before starting FastAPI
RUN make all

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI after preparing everything
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]