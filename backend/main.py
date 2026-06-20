import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routes import auth_routes, ai_routes

load_dotenv()

app = FastAPI(title="HackIndia API")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://summary-lbeqsk5d0-harsh-vaishnav-s-projects.vercel.app",
    "https://summary-lbeqsk5d0-harsh-vaishnav-s-projects.vercel.app/",
    FRONTEND_URL,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router, prefix="/auth", tags=["auth"])
app.include_router(ai_routes.router, prefix="/api", tags=["api"])

@app.get("/")
def health_check():
    return {"status": "HackIndia API running"}

@app.on_event("startup")
def startup_event():
    # Pre-load the model during startup so the first query doesn't timeout
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../summary.ai')))
    try:
        from embedding.embedder import _get_text_model
        print("Pre-loading SentenceTransformer model...")
        _get_text_model()
        print("Model pre-loaded successfully!")
    except Exception as e:
        print(f"Failed to pre-load model: {e}")
