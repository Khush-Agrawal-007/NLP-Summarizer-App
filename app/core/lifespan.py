# app/core/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
import nltk

from app.services.summarizer import SummarizerService

# NLTK resources required at runtime
NLTK_RESOURCES = [
    'tokenizers/punkt',
    'corpora/stopwords',
    'tokenizers/punkt_tab',
    'taggers/averaged_perceptron_tagger',
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("Starting up...")

    # Download any missing NLTK resources
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            name = resource.split('/')[-1]
            print(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

    # Load the summarizer model and store it in app state
    app.state.summarizer = SummarizerService()
    print("Models loaded.")

    yield

    # --- Shutdown ---
    print("Shutting down...")
    app.state.summarizer = None