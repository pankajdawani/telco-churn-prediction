# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import joblib
import json
import pandas as pd
import numpy as np

# Load model properly
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')

# Load scaler properly
scaler = joblib.load('scaler.pkl')

# Load features
with open('features.json', 'r') as f:
    features = json.load(f)

# Threshold
threshold = 0.409

# Initialize API
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using XGBoost",
    version="2.0"
)

# Input schema
class CustomerData(BaseModel):
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    high_value_at_risk: int
    num_services: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int
    tenure_group_Established: int
    tenure_group_Loyal: int
    tenure_group_New: int

# Health check
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "XGBoost Churn Predictor",
        "version": "2.0",
        "threshold": threshold
    }

# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert to dataframe
    data = pd.DataFrame([customer.dict()])

    # Rename columns to match training features
    column_mapping = {
        'InternetService_Fiber_optic': 'InternetService_Fiber optic',
        'Contract_One_year': 'Contract_One year',
        'Contract_Two_year': 'Contract_Two year',
        'PaymentMethod_Credit_card': 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic_check': 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed_check': 'PaymentMethod_Mailed check'
    }
    data = data.rename(columns=column_mapping)

    # Reorder columns to match training
    data = data[features]

    # Scale numerical features
    data[['tenure', 'MonthlyCharges']] = scaler.transform(
        data[['tenure', 'MonthlyCharges']]
    )

    # Get probability
    churn_prob = model.predict_proba(data)[0][1]

    # Apply threshold
    churn_prediction = int(churn_prob >= threshold)

    # Risk level
    if churn_prob >= 0.7:
        risk_level = "HIGH"
        recommendation = "Immediate intervention required — offer contract upgrade or discount"
    elif churn_prob >= 0.4:
        risk_level = "MEDIUM"
        recommendation = "Monitor closely — consider proactive outreach"
    else:
        risk_level = "LOW"
        recommendation = "Low risk — standard monitoring"

    return {
        "churn_probability": round(float(churn_prob), 4),
        "churn_prediction": churn_prediction,
        "risk_level": risk_level,
        "threshold_used": threshold,
        "recommendation": recommendation
    }