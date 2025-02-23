from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("best_rf_model.pkl")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],  
)

# Mount the static directory to serve HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve index.html
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# Load expected feature order (from training data)
expected_features = joblib.load("processed_data.pkl")[0].columns.tolist()
print("📌 Expected Feature Names (Order Must Match):")
print(expected_features)

# Define the input schema
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

@app.post("/predict")
async def predict(data: InputData):
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

    # Return the prediction result
    return {"prediction_mbps": prediction_mbps}