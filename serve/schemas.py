"""Pydantic request/response models for the FastAPI serving layer."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    description: str = Field(..., description="Transaction description / merchant string")
    amount: float = Field(..., description="Transaction amount")
    date: str = Field(..., description="Transaction date, ISO format YYYY-MM-DD")


class PredictResponse(BaseModel):
    category: str
    subcategory: str
    confidence: float
    category_confidence: float
    subcategory_confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
