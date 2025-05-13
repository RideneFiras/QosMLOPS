from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import joblib

import shap
import os
import time
from services.chatgpt_service import generate_qos_insight
from services.preprocessing import preprocess_single_row

# ✅ Environment detection
IS_DOCKER = os.getenv("IS_DOCKER", "false").lower() == "true"
POSTGRES_HOST = "db" if IS_DOCKER else "localhost"
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "predictions_db")

print(
    f"🔧 Running in {'Docker' if IS_DOCKER else 'Local'} mode — "
    f"Connecting to {POSTGRES_USER}@{POSTGRES_HOST}/{POSTGRES_DB}"
)

# ✅ Database setup
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(Integer)
    prediction_mbps = Column(Float)


Base.metadata.create_all(bind=engine)

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Load model & SHAP
model = joblib.load("Models/best_xgb_model.pkl")
explainer = shap.Explainer(model)

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.get("/index")
async def serve_alt_index():
    return FileResponse("static/index1.html")


@app.get("/explain-page")
async def serve_explain_page():
    return FileResponse("static/explain.html")


# ✅ DB session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ✅ Predict from raw input (unprocessed row)
@app.post("/predict")
async def predict(data: dict, db: Session = Depends(get_db)):
    input_df = preprocess_single_row(data)
    print("🔹 Processed input for prediction:")
    print(input_df)

    prediction = model.predict(input_df)
    prediction_mbps = float(prediction[0]) / 1e6

    new_prediction = Prediction(
        timestamp=int(time.time()), prediction_mbps=prediction_mbps
    )
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)

    return {"prediction_mbps": prediction_mbps}


# ✅ SHAP Explanation
@app.post("/explain")
async def explain(data: dict):
    input_df = preprocess_single_row(data)
    shap_values = explainer(input_df)

    explanation_mbps = [val / 1e6 for val in shap_values[0].values.tolist()]
    base_value_mbps = shap_values.base_values[0] / 1e6
    predicted_throughput_mbps = base_value_mbps + sum(explanation_mbps)

    return {
        "throughput_mbps": predicted_throughput_mbps,
        "explanation": explanation_mbps,
        "features": input_df.columns.tolist(),
        "base_value": base_value_mbps,
    }


# ✅ SHAP → GPT Insight
@app.post("/chat")
async def chat_explanation(data: dict):
    insight = generate_qos_insight(
        data["throughput_mbps"], data["explanation"], data["features"]
    )
    print("✅ Insight generated:\n", insight)
    return {"insight": insight}


# ✅ List all predictions
@app.get("/predictions")
async def get_predictions(db: Session = Depends(get_db)):
    return db.query(Prediction).all()
