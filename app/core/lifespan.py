# app/core/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
import nltk
from app.services.summarizer import SummarizerService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load resources
    print("Starting up and loading models...")
    
    # Download NLTK data
    try:
        resources = ['tokenizers/punkt', 'corpora/stopwords', 'tokenizers/punkt_tab', 'taggers/averaged_perceptron_tagger']
        for res in resources:
            try:
                nltk.data.find(res)
            except LookupError:
                # Extract resource name from path
                name = res.split('/')[-1]
                print(f"Downloading missing NLTK resource: {name}")
                nltk.download(name)
    except Exception as e:
        print(f"Warning: NLTK resource download failed: {e}")
        
    # Initialize and load the summarizer model
    summarizer_service = SummarizerService()
    app.state.summarizer = summarizer_service
    print("Models loaded successfully!")
    
    yield
    
    # Shutdown: Clean up resources
    print("Shutting down and clearing resources...")
    app.state.summarizer = None
    # Add any additional cleanup logic here