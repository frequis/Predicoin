"""API de inferência: carrega models/model.joblib e prevê o fechamento do BTC
no dia seguinte a partir dos últimos 7 fechamentos."""

import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, PositiveFloat

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/model.joblib"))
N_LAGS = 7

app = FastAPI(title="Predicoin")
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    # Do mais antigo ao mais recente; tamanho diferente de 7 ou valor <= 0 gera 422
    last_closes: list[PositiveFloat] = Field(min_length=N_LAGS, max_length=N_LAGS)


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # O modelo prevê o log-retorno de amanhã; converte de volta para preço
    log_retorno = model.predict([req.last_closes])[0]
    prediction = req.last_closes[-1] * np.exp(log_retorno)
    return PredictResponse(prediction=float(prediction))
