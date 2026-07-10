"""FastAPI serving layer. Depends only on the Predictor Protocol (src/predictor.py),
so swapping SklearnPredictor for a future TransformerPredictor is a one-line change
here with no endpoint/schema changes.

Run from the repo root (so `src`/`serve` imports resolve without an install step):
  uv run uvicorn serve.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from serve.schemas import HealthResponse, PredictRequest, PredictResponse
from src.predictor import SklearnPredictor

predictor: SklearnPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = SklearnPredictor()
    yield
    predictor = None


app = FastAPI(title="Transaction Classifier", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        result = predictor.predict(request.description, request.amount, request.date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(
        category=result.category,
        subcategory=result.subcategory,
        confidence=result.confidence,
        category_confidence=result.category_confidence,
        subcategory_confidence=result.subcategory_confidence,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded = predictor is not None and predictor.health_check()
    return HealthResponse(status="ok" if model_loaded else "unhealthy", model_loaded=model_loaded)
