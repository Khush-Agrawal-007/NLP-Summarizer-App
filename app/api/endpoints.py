# app/api/endpoints.py
import traceback
from fastapi import APIRouter, Request, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, List

from app.services.summarizer import SummarizerService
from app.services.technical_summarizer import TechnicalSummarizer
from app.utils.file_handler import extract_text_from_pdf
from app.core.config import settings
from app.evaluation.metrics import SummarizationEvaluator, format_evaluation_report
from .schemas import (
    SummarizationResponse,
    ComparisonResponse,
    TechnicalSummaryResponse,
    ProductComparisonResponse,
    EvaluationResponse,
    TrainingRequest,
    TrainingResponse,
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# --- Dependency helpers ---

def get_summarizer_service(request: Request) -> SummarizerService:
    """Return the summarizer service stored in app state at startup."""
    return request.app.state.summarizer


def get_technical_summarizer(request: Request) -> TechnicalSummarizer:
    """Return (or lazily create) the technical summarizer stored in app state."""
    if not hasattr(request.app.state, 'technical_summarizer'):
        request.app.state.technical_summarizer = TechnicalSummarizer()
    return request.app.state.technical_summarizer


# --- UI ---

@router.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Serve the main UI page."""
    return templates.TemplateResponse(request=request, name="index.html")


# --- Summarization ---

@router.post("/summarize", response_model=SummarizationResponse)
async def summarize(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    method: str = Form(...),
    summarizer: SummarizerService = Depends(get_summarizer_service),
):
    """Summarize text using the abstractive or extractive method."""
    try:
        if file and file.filename.endswith('.pdf'):
            input_text = extract_text_from_pdf(await file.read())
        elif text:
            input_text = text
        else:
            raise HTTPException(status_code=400, detail="Please provide either a PDF file or text input")

        if not input_text or len(input_text.strip()) < settings.MIN_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text is too short to summarize (minimum {settings.MIN_TEXT_LENGTH} characters)",
            )

        if method == "abstractive":
            summary = summarizer.abstractive_summarize(input_text)
        elif method == "extractive":
            summary = summarizer.extractive_summarize(input_text)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Choose 'abstractive' or 'extractive'")

        return SummarizationResponse(
            summary=summary,
            method=method,
            original_length=len(input_text),
            summary_length=len(summary),
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- Document Comparison ---

@router.post("/compare", response_model=ComparisonResponse)
async def compare_documents(
    files: Optional[List[UploadFile]] = File(None),
    texts: Optional[List[str]] = Form(None),
    method: str = Form("abstractive"),
    summarizer: SummarizerService = Depends(get_summarizer_service),
):
    """Compare multiple documents by summarizing each and finding shared themes."""
    try:
        documents = []

        if files:
            for file in files:
                if file.filename and file.filename.endswith('.pdf'):
                    text = extract_text_from_pdf(await file.read())
                    documents.append({"name": file.filename, "text": text})

        if texts:
            for i, text in enumerate(texts):
                if text and text.strip():
                    documents.append({"name": f"Document {i+1}", "text": text.strip()})

        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 documents to compare")
        if len(documents) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 documents allowed for comparison")

        summaries = []
        for doc in documents:
            if len(doc["text"].strip()) < settings.MIN_TEXT_LENGTH:
                raise HTTPException(status_code=400, detail=f"{doc['name']} is too short to summarize")

            summary = (
                summarizer.abstractive_summarize(doc["text"])
                if method == "abstractive"
                else summarizer.extractive_summarize(doc["text"])
            )
            summaries.append({"name": doc["name"], "summary": summary, "length": len(doc["text"])})

        analysis = summarizer.compare_summaries([s["summary"] for s in summaries])

        return ComparisonResponse(
            summaries=summaries,
            common_themes=analysis["common_themes"],
            unique_points=analysis["unique_points"],
            method=method,
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- Technical Summarization ---

@router.post("/technical-summarize", response_model=TechnicalSummaryResponse)
async def technical_summarize(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    tech_summarizer: TechnicalSummarizer = Depends(get_technical_summarizer),
):
    """Generate a structured technical summary (specs, category, price range)."""
    try:
        if file and file.filename.endswith('.pdf'):
            input_text = extract_text_from_pdf(await file.read())
        elif text:
            input_text = text
        else:
            raise HTTPException(status_code=400, detail="Please provide either a PDF file or text input")

        if not input_text or len(input_text.strip()) < settings.MIN_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text is too short to summarize (minimum {settings.MIN_TEXT_LENGTH} characters)",
            )

        result = tech_summarizer.extract_structured_summary(input_text)

        return TechnicalSummaryResponse(
            product_name=result['product_name'],
            category=result['category'],
            summary=result['summary'],
            key_specs=result['key_specs'],
            price_range=result['price_range'],
            original_length=len(input_text),
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- Product Comparison ---

@router.post("/compare-products", response_model=ProductComparisonResponse)
async def compare_products(
    files: Optional[List[UploadFile]] = File(None),
    texts: Optional[List[str]] = Form(None),
    tech_summarizer: TechnicalSummarizer = Depends(get_technical_summarizer),
):
    """Compare multiple technical products with side-by-side spec analysis."""
    try:
        documents = []

        if files:
            for file in files:
                if file.filename and file.filename.endswith('.pdf'):
                    documents.append(extract_text_from_pdf(await file.read()))

        if texts:
            for text in texts:
                if text and text.strip():
                    documents.append(text.strip())

        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 products to compare")
        if len(documents) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 products allowed for comparison")

        product_summaries = []
        for doc in documents:
            if len(doc.strip()) < settings.MIN_TEXT_LENGTH:
                raise HTTPException(status_code=400, detail="One or more documents are too short")
            product_summaries.append(tech_summarizer.extract_structured_summary(doc))

        comparison = tech_summarizer.compare_products(product_summaries)
        return ProductComparisonResponse(**comparison)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- Evaluation ---

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_summary(
    reference_text: str = Form(...),
    generated_text: str = Form(...),
):
    """Evaluate summary quality using ROUGE, BLEU, and BARTScore metrics."""
    try:
        evaluator = SummarizationEvaluator()
        scores = evaluator.evaluate_text_summary(generated_text, reference_text)
        report = format_evaluation_report(scores)
        return EvaluationResponse(scores=scores, report=report)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- Removed endpoints (kept as stubs for backward compatibility) ---

@router.get("/dataset-info")
async def get_dataset_info():
    raise HTTPException(status_code=404, detail="Endpoint removed")


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    raise HTTPException(status_code=404, detail="Endpoint removed")