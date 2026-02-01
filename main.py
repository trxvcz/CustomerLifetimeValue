from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np


app = FastAPI(title="CLV Prediction API")

artifacts = joblib.load("RandomForestRegressor.joblib")
model = artifacts["model"]
encoder = artifacts["encoder"]
scaler = artifacts["scaler"]

class CustomerData(BaseModel):
    state: str
    response: str
    coverage: str
    education: str
    effective_to_date: str
    employment_status: str
    gender: str
    income: float
    location_code: str
    marital_status: str
    monthly_premium: float
    months_since_last_claim: float
    months_since_policy_inception: float
    number_of_open_complaints: float
    number_of_policies: float
    policy_type: str
    policy: str
    renew_offer_type: str
    sales_channel: str
    total_claim_amount: float
    vehicle_class: str
    vehicle_size: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=500, detail="Model nie został załadowany.")

    input_df = pd.DataFrame([data.model_dump()])
    mapper = {
            "state": "State",
            "response": "Response",
            "coverage": "Coverage",
            "education": "Education",
            "effective_to_date": "Effective To Date",
            "employment_status": "EmploymentStatus",
            "gender": "Gender",
            "income": "Income",
            "location_code": "Location Code",
            "marital_status": "Marital Status",
            "monthly_premium": "Monthly Premium Auto",
            "months_since_last_claim": "Months Since Last Claim",
            "months_since_policy_inception": "Months Since Policy Inception",
            "number_of_open_complaints": "Number of Open Complaints",
            "number_of_policies": "Number of Policies",
            "policy_type": "Policy Type",
            "policy": "Policy",
            "renew_offer_type": "Renew Offer Type",
            "sales_channel": "Sales Channel",
            "total_claim_amount": "Total Claim Amount",
            "vehicle_class": "Vehicle Class",
            "vehicle_size": "Vehicle Size"
    }
    input_df.rename(columns=mapper, inplace=True)


    required_cols = [
    'Customer'
    'State',
    'Response',
    'Coverage',
    'Education',
    'EmploymentStatus',
    'Gender',
    'Income',
    'LocationCode',
    'MaritalStatus',
    'MonthlyPremiumAuto',
    'MonthsSinceLastClaim',
    'MonthsSincePolicyInception',
    'NumberofOpenComplaints',
    'NumberofPolicies',
    'PolicyType',
    'Policy',
    'RenewOfferType',
    'SalesChannel',
    'TotalClaimAmount',
    'VehicleClass',
    'VehicleSize',

    ]

    for col in required_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    if 'Customer Lifetime Value' in input_df.columns:
        input_df = input_df.drop(columns=['Customer Lifetime Value'])

    try:
        transformed_df = encoder.transform(input_df)
        scaled_data = scaler.transform(transformed_df)
        prediction_log = model.predict(scaled_data)
        prediction = np.exp(prediction_log)

        return {"prediction_clv": float(prediction[0]),"currency":"USD"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
