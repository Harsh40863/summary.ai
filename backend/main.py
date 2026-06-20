import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from routes import auth_routes, ai_routes
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()

app = FastAPI(title="HackIndia API")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# ─── Force CORS headers on EVERY response (including 500/502 errors) ──────────
# This middleware runs BEFORE any route handler so even crash responses get headers
class ForceCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Handle preflight OPTIONS immediately
        if request.method == "OPTIONS":
            response = JSONResponse(content={}, status_code=200)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response

        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                content={"detail": "Internal server error"},
                status_code=500
            )

        # Always inject CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
        return response

# Add force-CORS middleware FIRST (outermost layer)
app.add_middleware(ForceCORSMiddleware)

# Standard CORS middleware as backup
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

@app.get("/ping")
def ping():
    """Lightweight keep-alive endpoint — called by frontend to prevent cold starts"""
    return {"pong": True}

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
