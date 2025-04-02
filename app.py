from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import shap


# ✅ Determine environment and configure PostgreSQL host
IS_DOCKER = os.getenv("IS_DOCKER", "false").lower() == "true"
POSTGRES_HOST = "db" if IS_DOCKER else "localhost"
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "predictions_db")

print(
    f"🔧 Running in {'Docker' if IS_DOCKER else 'Local'} mode — "
    f"Connecting to {POSTGRES_USER}@{POSTGRES_HOST}/{POSTGRES_DB}"
)

# ✅ Build the DATABASE_URL
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"

# ✅ Connect to PostgreSQL
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ✅ Define Prediction Table
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(Integer)
    prediction_mbps = Column(Float)


# ✅ Create Database Tables
Base.metadata.create_all(bind=engine)

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Load the trained model
model = joblib.load("best_rf_model.pkl")

# Initiliaze SHAP
explainer = shap.Explainer(model)

# ✅ Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount the static directory to serve HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")


# ✅ Route to serve index.html
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/explain-page")
async def serve_explain_page():
    return FileResponse("static/explain.html")


# ✅ Dependency to Get DB Session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ✅ Load expected feature order (from training data)
expected_features = joblib.load("processed_data.pkl")[0].columns.tolist()
print("📌 Expected Feature Names (Order Must Match):")
print(expected_features)


# ✅ Define the input schema
class InputData(BaseModel):
    timestamp: int
    PCell_RSRP_max: float
    PCell_RSRQ_max: float
    PCell_RSSI_max: float
    PCell_SNR_max: float
    PCell_Downlink_Num_RBs: float
    PCell_Downlink_Average_MCS: float
    PCell_Downlink_bandwidth_MHz: float
    PCell_Cell_Identity: float
    PCell_freq_MHz: float
    SCell_RSRP_max: float
    SCell_RSRQ_max: float
    SCell_RSSI_max: float
    SCell_SNR_max: float
    SCell_Downlink_Num_RBs: float
    SCell_Downlink_Average_MCS: float
    SCell_Downlink_bandwidth_MHz: float
    SCell_Cell_Identity: float
    SCell_freq_MHz: float
    operator: int
    Latitude: float
    Longitude: float
    Altitude: float
    speed_kmh: float
    COG: float
    precipIntensity: float
    precipProbability: float
    temperature: float
    apparentTemperature: float
    dewPoint: float
    humidity: float
    pressure: float
    windSpeed: float
    cloudCover: float
    uvIndex: float
    visibility: float
    Traffic_Jam_Factor: float
    area: int  # Remember, 'area' was encoded!


# ✅ Prediction Endpoint
@app.post("/predict")
async def predict(data: InputData, db: Session = Depends(get_db)):
    # Convert input data to a dictionary
    input_dict = data.dict()

    # ✅ Rename "Traffic_Jam_Factor" to "Traffic Jam Factor"
    input_dict["Traffic Jam Factor"] = input_dict.pop("Traffic_Jam_Factor")

    # ✅ Ensure feature order matches the trained model
    input_df = pd.DataFrame([input_dict])[expected_features]  # Force column order

    # Log actual input for debugging
    print("🔹 Received Input for Prediction:")
    print(input_df)

    # Make a prediction
    prediction = model.predict(input_df)

    # ✅ Convert to Megabits per second (Mbit/s)
    prediction_mbps = float(prediction[0]) / 1e6

    # ✅ Save to PostgreSQL Database
    new_prediction = Prediction(
        timestamp=data.timestamp, prediction_mbps=prediction_mbps
    )
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)

    # Return the prediction result
    return {"prediction_mbps": prediction_mbps}


# ✅ Endpoint to Get All Predictions
@app.get("/predictions")
async def get_predictions(db: Session = Depends(get_db)):
    predictions = db.query(Prediction).all()
    return predictions


@app.post("/explain")
async def explain(data: InputData):
    input_dict = data.dict()
    input_dict["Traffic Jam Factor"] = input_dict.pop("Traffic_Jam_Factor")
    input_df = pd.DataFrame([input_dict])[expected_features]

    # Generate SHAP values
    shap_values = explainer(input_df)

    # Convert SHAP values to Mbps (model predicts in bps)
    explanation_mbps = [val / 1e6 for val in shap_values[0].values.tolist()]
    base_value_mbps = shap_values.base_values[0] / 1e6
    predicted_throughput_mbps = base_value_mbps + sum(explanation_mbps)

    # Return cleaned explanation
    return {
        "throughput_mbps": predicted_throughput_mbps,
        "explanation": explanation_mbps,
        "features": input_df.columns.tolist(),
        "base_value": base_value_mbps,
    }


@app.post("/chat")
async def chat_explanation(data: dict):
    throughput = data["throughput_mbps"]
    shap_values = data["explanation"]
    features = data["features"]

    # Format SHAP features list
    formatted_features = "\n".join(
        [
            f"- {feat}: {'+' if val >= 0 else ''}{round(val, 3)} Mbps"
            for feat, val in zip(features, shap_values)
        ]
    )

    # Full prompt
    prompt = f"""
You are a telecom and AI expert assisting a 5G network optimization system.

A machine learning model has predicted a user's throughput in Mbps based on 5G network features.
It also provides SHAP values showing how each feature influenced the predicted speed.

📡 Predicted Throughput: {round(throughput, 2)} Mbps

📊 SHAP Feature Contributions:
{formatted_features}

Your task is to analyze this prediction and provide a clear explanation in the context of Quality of Service (QoS).

Please include:
- **QoS Rating**
  Categorize the throughput into one of the following:
  - Very Low (<15 Mbps)
  - Low (15–30 Mbps)
  - Medium (30–60 Mbps)
  - Good (60–100 Mbps)
  - Very Good (>100 Mbps)

- **Top Influencing Factors**
  Summarize what increased or decreased the throughput.
  Use terms like RSRP, SNR, bandwidth, congestion, interference, etc.

- **Recommendations**
  If QoS is Low or Very Low → suggest improvements (e.g., optimize SNR, boost resource allocation).
  If QoS is Good or Very Good → explain what worked well.

✅ Use a professional but simple tone.
❌ Do not list raw SHAP numbers — summarize the insights clearly.
""".strip()

    return {"prompt_preview": prompt}
