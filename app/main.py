# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.core.lifespan import lifespan

app = FastAPI(
    title="Text Summarizer API",
    lifespan=lifespan
)

# Allow all origins (suitable for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register all API routes
app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}