import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import ai_routes, auth_routes

env_path = Path(__file__).resolve().parent / ".env"
root_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=root_env_path, override=False)

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

app = FastAPI(title="HackIndia Secure API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router, prefix="/auth", tags=["auth"])
app.include_router(ai_routes.router, prefix="/api", tags=["ai"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
