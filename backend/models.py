from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------
# Authentication Models
# ---------------------------------------------------------

class UserSignup(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------------------------------------------------------
# AI Generation Models
# ---------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    result: str


# ---------------------------------------------------------
# Document Search & RAG Query Models
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    action: str = Field(..., description="Action type: search, explore, or think")
    threshold: Optional[float] = Field(0.35, description="Similarity threshold for relevance")
    translate_to: Optional[str] = Field("en", description="Target language code (e.g. en, hi, fr, es, de, ja, ko, zh)")


class TranslationInfo(BaseModel):
    target_language: str
    target_language_name: str
    source_language: str
    translated: bool


class QueryResponse(BaseModel):
    success: bool
    message: str
    results: List[Dict[str, Any]]
    action: str
    timestamp: str
    query: str
    translation: Optional[TranslationInfo] = None


# ---------------------------------------------------------
# Document Upload & Management Models
# ---------------------------------------------------------

class UploadResponse(BaseModel):
    success: bool
    message: str
    uploaded_files: List[Dict[str, Any]]
    timestamp: str


class DocumentInfo(BaseModel):
    id: str
    name: str
    content_length: int
    upload_date: str


class DocumentsResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]
    clusters: int
    timestamp: str


# ---------------------------------------------------------
# System Status & Utility Models
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    has_documents: bool
    has_embeddings: bool
    has_search_engine: bool
    document_count: int
    timestamp: str


class SupportedLanguagesResponse(BaseModel):
    languages: Dict[str, str]
    default: str
    note: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None
    timestamp: str
