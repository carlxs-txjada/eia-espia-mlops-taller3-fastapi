from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import pandas as pd
import joblib

# Cargar el pipeline que integra preprocesamiento + modelo
model = joblib.load("pipeline.joblib")

app = FastAPI(
    title="eia-espia-mlops-taller3 API",
    description="Endpoint para inferencia con Pipeline de ML",
    version="1.0.0"
)

class DataPredict(BaseModel):
    data_to_predict: list[list] = [
        [23, "male", 34.4, 0, "no", "southwest"],
        [35, "female", 40.0, 1, "no", "northwest"]
    ]

@app.get("/")
def home():
    return {
        "project": "eia-espia-mlops-taller3",
        "institution": "Universidad EIA",
        "status": "active"
    }

@app.post("/predict")
def predict(request: DataPredict):
    try:
        columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        df_input = pd.DataFrame(request.data_to_predict, columns=columns)
        predictions = model.predict(df_input)
        return {"predictions": predictions.tolist()}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

