# app/utils/file_handler.py
import io
import PyPDF2
from fastapi import HTTPException


def extract_text_from_pdf(file_content: bytes) -> str:
    """Read all pages of a PDF and return the combined plain text."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        pages = [page.extract_text() for page in reader.pages]
        return "\n".join(p for p in pages if p).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")