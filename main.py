from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="UrbanFlux Spoilage Prediction API")

# ===== Load ML Models =====
model = joblib.load("spoilage_model.pkl")
product_encoder = joblib.load("product_encoder.pkl")
packaging_encoder = joblib.load("packaging_encoder.pkl")

# ===== Secure API key from environment =====
API_KEY = os.getenv("API_KEY")


# ===== Input Schema =====
class ShipmentData(BaseModel):
    product_type: str
    initial_quality: float
    packaging_type: str
    temperature: float
    humidity: float
    travel_time: float
    delay_time: float
    distance: float
    shelf_life: float


# ===== Prediction Endpoint =====
@app.post("/predict")
def predict_spoilage(shipment: ShipmentData, api_key: str = Header(None)):

    # 🔐 API Key Security
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Convert input to dataframe
        df = pd.DataFrame([shipment.dict()])

        # Encode categorical columns
        df["product_type"] = product_encoder.transform(df["product_type"])
        df["packaging_type"] = packaging_encoder.transform(df["packaging_type"])

        # Feature Engineering
        df["storage_duration"] = df["travel_time"] + df["delay_time"]
        df["transit_stress"] = df["temperature"] * df["storage_duration"]

        # Training feature list (must match training exactly)
        features = [
            "product_type",
            "initial_quality",
            "packaging_type",
            "temperature",
            "humidity",
            "storage_duration",
            "transit_stress",
            "shelf_life"
        ]

        # ⭐ IMPORTANT FIX — rebuild dataframe with feature names
        X = pd.DataFrame(df[features].values, columns=features)

        # Predict probability
        risk_prob = model.predict_proba(X)[0][1]

        # Risk level logic
        if risk_prob > 0.7:
            risk_level = "HIGH"
        elif risk_prob > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "spoilage_risk_probability": round(float(risk_prob), 3),
            "risk_level": risk_level
        }

    except Exception as e:
        return {"error": str(e)}
