from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Any, Dict

app = FastAPI(
    title="Modelo de Clasificación de Clientes Bancarios",
    description="API para predecir si un cliente bancario suscribirá un depósito a plazo fijo.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str         
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    contacted_before: float

    class Config:
        json_schema_extra = {
            "example": {               
                "age": 35,
                "job": "technician",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 200,
                "campaign": 1,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": -1.8,
                "cons_price_idx": 92.893,
                "cons_conf_idx": -46.2,
                "euribor3m": 1.299,
                "nr_employed": 5099.1,
                "contacted_before": 0.0
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    probability: Dict[str, float]   # FIX: era Dict[float, Any]
    model_info: Dict[str, Any]


MODEL_PATH = "models/decision_tree_model.pkl"

try:
    model = joblib.load(MODEL_PATH)   
except FileNotFoundError as e:
    model = None
    raise RuntimeError(f"No se pudo cargar el modelo desde {MODEL_PATH}: {e}")


@app.get("/")
def read_root():
    return {
        "message": "API del modelo de clasificación de clientes bancarios está en funcionamiento."
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Realiza una predicción utilizando el modelo cargado."""

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo no está cargado."
        )

    try:
        input_data = pd.DataFrame([request.dict()])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        class_labels = model.classes_
        probability_dict = {
            str(class_labels[i]): float(probability[i])  
            for i in range(len(class_labels))
        }

        model_info = {
            "model_type": type(model).__name__
        }

        return PredictionResponse(
            prediction=str(prediction),
            probability=probability_dict,
            model_info=model_info
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al realizar la predicción: {str(e)}"
        )