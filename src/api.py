from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, Depends
from pydantic import BaseModel, ConfigDict

from src.predict import Predictor
from src.database import Database

class BankNoteFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variance: float
    skewness: float
    curtosis: float
    entropy: float

class PredictionResponse(BaseModel):
    prediction: int
    label: str

app = FastAPI(
    title="Banknote Authentication API",
    description="API for classifying banknotes as authentic or forged"
)

@lru_cache()
def get_predictor() -> Predictor:
    return Predictor()

@lru_cache()
def get_database() -> Database:
    database = Database()
    database.initialize()
    return database

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(
        features: BankNoteFeatures,
        predictor: Annotated[Predictor, Depends(get_predictor)],
        database: Annotated[Database, Depends(get_database)]
) -> PredictionResponse:
    features_data = features.model_dump()

    predicted_class = predictor.predict(
        features.model_dump()
    )

    class_names = {
        0: "authentic",
        1: "forged"
    }

    label = class_names[predicted_class]

    database.save_prediction(
        features=features_data,
        prediction=predicted_class,
        label=label
    )

    return PredictionResponse(
        prediction=predicted_class,
        label=label
    )