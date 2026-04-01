# app/api/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SummarizationResponse(BaseModel):
    summary: str
    method: str
    original_length: int
    summary_length: int


class ComparisonResponse(BaseModel):
    summaries: List[Dict[str, Any]]
    common_themes: List[str]
    unique_points: Dict[str, List[str]]
    method: str


# pros, cons, and best_for are intentionally excluded from the API response
class TechnicalSummaryResponse(BaseModel):
    product_name: str
    category: str
    summary: str
    key_specs: Dict[str, str]
    price_range: str
    original_length: int


class ProductComparisonResponse(BaseModel):
    product_count: int
    same_category: bool
    category: str
    products: List[Dict[str, str]]
    spec_comparison: Dict[str, List[Dict[str, str]]]
    summary: str


class EvaluationResponse(BaseModel):
    scores: Dict[str, float]
    report: str


class TrainingRequest(BaseModel):
    num_epochs: int = 3
    batch_size: int = 2
    model_name: str = "facebook/bart-large-cnn"


class TrainingResponse(BaseModel):
    status: str
    message: str
    metrics: Optional[Dict[str, Any]] = None