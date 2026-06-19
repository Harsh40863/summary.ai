import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is missing!")

def generate_answer_with_google(query, summary, doc_name, action_type="search"):
    context = f"From document '{doc_name}':\n{summary}\n"
    
    if action_type == "think":
        prompt = (
            f"You are an expert analytical assistant. The user asked: '{query}'.\n\n"
            f"Using only the context provided below, perform a deep, comprehensive analysis. "
            f"Draw underlying insights, evaluate implications, and explain the 'why' and 'how'. "
            f"Structure your response beautifully with clear paragraphs and markdown bullet points to make it highly readable.\n\n"
            f"{context}\nAnswer:"
        )
    else:
        prompt = (
            f"You are a helpful assistant. The user asked: '{query}'.\n\n"
            f"Using only the context provided below, provide a highly precise, factual, and direct answer. "
            f"Do not add fluff, just extract the relevant information clearly using bullet points if needed.\n\n"
            f"{context}\nAnswer:"
        )

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        response = requests.post(
            url,
            params={"key": GOOGLE_API_KEY},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.8,
                    "maxOutputTokens": 8192,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error generating response: {e}"
