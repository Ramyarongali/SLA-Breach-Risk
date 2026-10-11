
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap

app = FastAPI(title="SLA Breach Prediction API")

model = joblib.load("sla_breach_model.pkl")

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(xgb_model)


class IncidentData(BaseModel):
    impact: str
    urgency: str
    priority: str
    reassignment_count: int
    reopen_count: int
    sys_mod_count: int
    knowledge: int
    u_priority_confirmation: int
    category: str
    subcategory: str
    contact_type: str
    assignment_group: str


def prepare_input(data):
    return pd.DataFrame([{
        "impact": data.impact,
        "urgency": data.urgency,
        "priority": data.priority,
        "reassignment_count": data.reassignment_count,
        "reopen_count": data.reopen_count,
        "sys_mod_count": data.sys_mod_count,
        "knowledge": data.knowledge,
        "u_priority_confirmation": data.u_priority_confirmation,
        "category": data.category,
        "subcategory": data.subcategory,
        "contact_type": data.contact_type,
        "assignment_group": data.assignment_group
    }])


@app.get("/")
def home():
    return {"message": "SLA Breach Prediction API is running"}


@app.post("/predict")
def predict(data: IncidentData):
    input_data = prepare_input(data)

    probability = model.predict_proba(input_data)[0][1]
    breach_risk = round(float(probability) * 100, 2)

    transformed = preprocessor.transform(input_data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_values = explainer.shap_values(transformed)

    values = np.asarray(shap_values)

    if values.ndim == 3:
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]

    contributions = []

    for name, value in zip(feature_names, values):
        contributions.append({
            "feature": str(name),
            "contribution": round(float(value), 5)
        })

    contributions.sort(
        key=lambda item: abs(item["contribution"]),
        reverse=True
    )

    top_factors = []

    for item in contributions[:5]:
        top_factors.append({
            "feature": item["feature"],
            "contribution": item["contribution"],
            "effect": (
                "increases predicted breach risk"
                if item["contribution"] > 0
                else "decreases predicted breach risk"
            )
        })

    return {
        "breach_risk": breach_risk,
        "risk_level": (
            "High" if breach_risk >= 60
            else "Medium" if breach_risk >= 30
            else "Low"
        ),
        "explanation_method": "SHAP",
        "top_factors": top_factors
    }
