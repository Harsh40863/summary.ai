import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.auth import get_current_user

env_path = Path(__file__).resolve().parent.parent / ".env"
root_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=root_env_path, override=False)

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str


@router.post("/generate")
def generate(payload: GenerateRequest, current_user: str = Depends(get_current_user)):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GEMINI_API_KEY is not configured",
        )

    model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    try:
        response = requests.post(
            url,
            params={"key": gemini_api_key},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": payload.prompt}],
                    }
                ]
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        result = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"result": result}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API call failed: {exc}",
        ) from exc
