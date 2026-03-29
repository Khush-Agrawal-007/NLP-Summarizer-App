# app/api/endpoints.py
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
    TrainingResponse
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def get_summarizer_service(request: Request) -> SummarizerService:
    """Dependency to get the summarizer service from app state."""
    return request.app.state.summarizer

def get_technical_summarizer(request: Request) -> TechnicalSummarizer:
    """Dependency to get the technical summarizer service from app state."""
    if not hasattr(request.app.state, 'technical_summarizer'):
        request.app.state.technical_summarizer = TechnicalSummarizer()
    return request.app.state.technical_summarizer

@router.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Serve the HTML UI"""
    return templates.TemplateResponse(request=request, name="index.html")

@router.post("/summarize", response_model=SummarizationResponse)
async def summarize(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    method: str = Form(...),
    summarizer: SummarizerService = Depends(get_summarizer_service)
):
    """
    Summarize text using either abstractive or extractive method
    """
    input_text = ""
    try:
        if file and file.filename.endswith('.pdf'):
            content = await file.read()
            input_text = extract_text_from_pdf(content)
        elif text:
            input_text = text
        else:
            raise HTTPException(status_code=400, detail="Please provide either a PDF file or text input")

        if not input_text or len(input_text.strip()) < settings.MIN_TEXT_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Text is too short to summarize (minimum {settings.MIN_TEXT_LENGTH} characters)"
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
            summary_length=len(summary)
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonResponse)
async def compare_documents(
    files: Optional[List[UploadFile]] = File(None),
    texts: Optional[List[str]] = Form(None),
    method: str = Form("abstractive"),
    summarizer: SummarizerService = Depends(get_summarizer_service)
):
    """
    Compare multiple documents by generating summaries and finding similarities/differences
    """
    try:
        documents = []
        
        # Process uploaded files
        if files:
            for file in files:
                if file.filename and file.filename.endswith('.pdf'):
                    content = await file.read()
                    text = extract_text_from_pdf(content)
                    documents.append({"name": file.filename, "text": text})
        
        # Process text inputs
        if texts:
            for i, text in enumerate(texts):
                if text and text.strip():
                    documents.append({"name": f"Document {i+1}", "text": text.strip()})
        
        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 documents to compare")
        
        if len(documents) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 documents allowed for comparison")
        
        # Generate summaries for each document
        summaries = []
        for doc in documents:
            if len(doc["text"].strip()) < settings.MIN_TEXT_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"{doc['name']} is too short to summarize"
                )
            
            if method == "abstractive":
                summary = summarizer.abstractive_summarize(doc["text"])
            else:
                summary = summarizer.extractive_summarize(doc["text"])
            
            summaries.append({
                "name": doc["name"],
                "summary": summary,
                "length": len(doc["text"])
            })
        
        # Find common themes and differences
        comparison_analysis = summarizer.compare_summaries([s["summary"] for s in summaries])
        
        return ComparisonResponse(
            summaries=summaries,
            common_themes=comparison_analysis["common_themes"],
            unique_points=comparison_analysis["unique_points"],
            method=method
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/technical-summarize", response_model=TechnicalSummaryResponse)
async def technical_summarize(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    tech_summarizer: TechnicalSummarizer = Depends(get_technical_summarizer)
):
    """
    Generate a structured technical summary with specs, pros, cons, etc.
    """
    input_text = ""
    try:
        if file and file.filename.endswith('.pdf'):
            content = await file.read()
            input_text = extract_text_from_pdf(content)
        elif text:
            input_text = text
        else:
            raise HTTPException(status_code=400, detail="Please provide either a PDF file or text input")

        if not input_text or len(input_text.strip()) < settings.MIN_TEXT_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Text is too short to summarize (minimum {settings.MIN_TEXT_LENGTH} characters)"
            )

        # Generate structured summary
        try:
            structured_summary = tech_summarizer.extract_structured_summary(input_text)
        except Exception as summary_error:
            print(f"Error generating structured summary: {summary_error}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating summary: {str(summary_error)}. Please try again with a different description."
            )
        
        return TechnicalSummaryResponse(
            product_name=structured_summary['product_name'],
            category=structured_summary['category'],
            summary=structured_summary['summary'],
            key_specs=structured_summary['key_specs'],
            pros=structured_summary['pros'],
            cons=structured_summary['cons'],
            best_for=structured_summary['best_for'],
            price_range=structured_summary['price_range'],
            original_length=len(input_text)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Technical summarization error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.post("/compare-products", response_model=ProductComparisonResponse)
async def compare_products(
    files: Optional[List[UploadFile]] = File(None),
    texts: Optional[List[str]] = Form(None),
    tech_summarizer: TechnicalSummarizer = Depends(get_technical_summarizer)
):
    """
    Compare multiple technical products with structured analysis
    """
    try:
        documents = []
        
        # Process uploaded files
        if files:
            for file in files:
                if file.filename and file.filename.endswith('.pdf'):
                    content = await file.read()
                    text = extract_text_from_pdf(content)
                    documents.append(text)
        
        # Process text inputs
        if texts:
            for text in texts:
                if text and text.strip():
                    documents.append(text.strip())
        
        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 products to compare")
        
        if len(documents) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 products allowed for comparison")
        
        # Generate structured summaries for each product
        product_summaries = []
        for doc in documents:
            if len(doc.strip()) < settings.MIN_TEXT_LENGTH:
                raise HTTPException(status_code=400, detail="One or more documents are too short")
            
            summary = tech_summarizer.extract_structured_summary(doc)
            product_summaries.append(summary)
        
        # Compare products
        comparison = tech_summarizer.compare_products(product_summaries)
        
        return ProductComparisonResponse(**comparison)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Product comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_summary(
    reference_text: str = Form(...),
    generated_text: str = Form(...)
):
    """
    Evaluate summary quality using ROUGE, BLEU, and BARTScore metrics
    """
    try:
        evaluator = SummarizationEvaluator()
        scores = evaluator.evaluate_text_summary(generated_text, reference_text)
        report = format_evaluation_report(scores)
        
        return EvaluationResponse(
            scores=scores,
            report=report
        )
    except Exception as e:
        print(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset-info")
async def get_dataset_info():
    raise HTTPException(status_code=404, detail="Endpoint removed")


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    raise HTTPException(status_code=404, detail="Endpoint removed")