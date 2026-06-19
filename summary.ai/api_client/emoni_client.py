import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is missing!")

# ✅ This now uses google-generativeai Gemini SDK (not v1beta!)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-001",  # This exact string is required!
    temperature=0.8,
    google_api_key=GOOGLE_API_KEY
)

def generate_answer_with_google(query, summary, doc_name):
    context = f"From document '{doc_name}':\n{summary}\n"
    prompt = (
        f"You are an expert assistant. The user asked: '{query}'.\n\n"
        f"Using only the context provided below, generate a detailed, thoughtful, and precise answer:\n\n"
        f"{context}\nAnswer:"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"❌ Error generating response: {e}"