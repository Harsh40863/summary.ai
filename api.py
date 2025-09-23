# ---------------------------
# FastAPI Application with Translation for Search and Think Only
# ---------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import traceback
from datetime import datetime
import os

# Import the logic module
from backend.logic import get_search_engine, reset_search_engine, DocumentSearchEngine

# Import translation functions
from translation.redis_translator import translate_response, get_supported_languages, is_supported_language

# ---------------------------
# Configure Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# FastAPI App Configuration
# ---------------------------
app = FastAPI(
    title="Smart AI-Powered Document Search API with Translation",
    description="API for intelligent document search, exploration, and analysis with translation support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ---------------------------
# CORS Configuration
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Updated Pydantic Models
# ---------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    action: str = Field(..., description="Action type: search, explore, think, or ppt")
    threshold: Optional[float] = Field(0.35, description="Similarity threshold for relevance")
    translate_to: Optional[str] = Field('en', description="Target language (en, hi, fr, es, de, ja, ko, zh) - Only for search and think actions")

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

class UploadResponse(BaseModel):
    success: bool
    message: str
    uploaded_files: List[Dict[str, Any]]
    timestamp: str

class DocumentInfo(BaseModel):
    name: str
    content_length: int
    upload_date: str

class DocumentsResponse(BaseModel):
    total_documents: int
    documents: List[DocumentInfo]
    clusters: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    has_documents: bool
    has_embeddings: bool
    has_search_engine: bool
    document_count: int
    timestamp: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None
    timestamp: str

class SupportedLanguagesResponse(BaseModel):
    languages: Dict[str, str]
    default: str
    note: str

# ---------------------------
# Dependency Functions
# ---------------------------
def get_engine() -> DocumentSearchEngine:
    """Dependency to get the search engine instance"""
    try:
        engine = get_search_engine()
        # Check if engine is properly initialized
        if engine.documents is None:
            logger.warning("Search engine initialized with empty state")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        # Create a minimal engine instance instead of failing completely
        try:
            engine = DocumentSearchEngine.__new__(DocumentSearchEngine)
            engine.documents = []
            engine.doc_texts = []
            engine.doc_embeddings = None
            engine.index_to_docinfo = {}
            engine.topic_labels = None
            engine.search_engine = None
            engine.n_clusters = 0
            return engine
        except Exception as fallback_error:
            logger.error(f"Fallback engine creation failed: {fallback_error}")
            raise HTTPException(status_code=500, detail="Search engine initialization failed")

# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart AI-Powered Document Search API with Translation",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "languages": "/languages"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(engine: DocumentSearchEngine = Depends(get_engine)):
    """Health check endpoint"""
    try:
        health_data = engine.health_check()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/languages", response_model=SupportedLanguagesResponse)
async def get_languages():
    """Get supported translation languages"""
    try:
        return SupportedLanguagesResponse(
            languages=get_supported_languages(),
            default="en",
            note="Translation available for 'search' and 'think' actions only. 'explore' action remains in original language."
        )
    except Exception as e:
        logger.error(f"Failed to get languages: {e}")
        raise HTTPException(status_code=500, detail="Failed to get supported languages")

@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(..., description="Files to upload (PDF/DOCX/PPTX/TXT/JPG/PNG)"),
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Upload multiple files to the system"""
    try:
        # Validate file types
        allowed_extensions = {'.pdf', '.docx', '.pptx', '.txt', '.jpg', '.jpeg', '.png'}
        
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400, 
                    detail="File must have a filename"
                )
            
            file_ext = os.path.splitext(file.filename.lower())[1]
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type not supported: {file.filename}. Allowed: PDF, DOCX, PPTX, TXT, JPG, PNG"
                )
        
        # Upload files
        upload_results = engine.upload_files(files)
        logger.info(f"Upload results: {upload_results}")
        
        success_count = sum(1 for result in upload_results if result.get('status') == 'success')
        
        return UploadResponse(
            success=success_count > 0,
            message=f"Successfully uploaded {success_count}/{len(files)} files",
            uploaded_files=upload_results,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Process a search query with specified action and optional translation"""
    try:
        # Validate action
        valid_actions = ['search', 'explore', 'think', 'ppt']
        if request.action.lower() not in valid_actions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid action '{request.action}'. Valid actions: {', '.join(valid_actions)}"
            )
        
        # Validate language if provided for translatable actions
        if (request.translate_to and 
            request.translate_to != 'en' and 
            request.action.lower() in ['search', 'think']):
            
            if not is_supported_language(request.translate_to):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported language: {request.translate_to}. Supported: {list(get_supported_languages().keys())}"
                )
        
        # Process the query
        result = engine.process_query(request.query, request.action, request.threshold)
        
        # Apply translation only for search and think actions
        if (request.translate_to and 
            request.translate_to != 'en' and 
            request.action.lower() in ['search', 'think']):
            
            logger.info(f"Translating response to {request.translate_to} for action {request.action}")
            result = translate_response(result, request.translate_to)
        
        if not result['success']:
            return QueryResponse(
                success=False,
                message=result['message'],
                results=[],
                action=request.action,
                timestamp=datetime.now().isoformat(),
                query=request.query,
                translation=TranslationInfo(**result['translation']) if 'translation' in result else None
            )
        
        return QueryResponse(
            success=True,
            message=result['message'],
            results=result['results'],
            action=request.action,
            timestamp=datetime.now().isoformat(),
            query=request.query,
            translation=TranslationInfo(**result['translation']) if 'translation' in result else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/query", response_model=QueryResponse)
async def process_query_get(
    q: str = Query(..., description="Search query"),
    action: str = Query("search", description="Action type: search, explore, or think"),
    threshold: float = Query(0.35, description="Similarity threshold"),
    translate_to: str = Query('en', description="Target language (only for search/think)"),
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Process a search query via GET request (for simple frontend integration)"""
    request = QueryRequest(query=q, action=action, threshold=threshold, translate_to=translate_to)
    return await process_query(request, engine)

@app.get("/documents", response_model=DocumentsResponse)
async def get_documents_info(engine: DocumentSearchEngine = Depends(get_engine)):
    """Get information about uploaded documents"""
    try:
        doc_info = engine.get_documents_info()
        
        return DocumentsResponse(
            total_documents=doc_info['total_documents'],
            documents=[DocumentInfo(**doc) for doc in doc_info['documents']],
            clusters=doc_info['clusters'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get documents info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents information")

@app.post("/documents/reload")
async def reload_documents(engine: DocumentSearchEngine = Depends(get_engine)):
    """Reload documents from database (useful after external uploads)"""
    try:
        engine.reload_documents()
        doc_info = engine.get_documents_info()
        
        return {
            "success": True,
            "message": "Documents reloaded successfully",
            "document_count": doc_info['total_documents'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reload documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload documents")

# ---------------------------
# Dedicated Endpoints with Translation Support
# ---------------------------

@app.post("/search", response_model=QueryResponse)
async def search_documents(
    request: QueryRequest,
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Dedicated search endpoint with translation support"""
    request.action = "search"
    return await process_query(request, engine)

@app.post("/explore", response_model=QueryResponse)
async def explore_documents(
    request: QueryRequest,
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Dedicated explore endpoint (NO translation - keeps original language)"""
    request.action = "explore"
    # Note: Translation is ignored for explore action
    return await process_query(request, engine)

@app.post("/think", response_model=QueryResponse)
async def think_documents(
    request: QueryRequest,
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Dedicated think endpoint with translation support"""
    request.action = "think"
    return await process_query(request, engine)

@app.post("/ppt", response_model=QueryResponse)
async def generate_ppt(
    request: QueryRequest,
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Generate a decorative PPT from query summary (no translation)."""
    request.action = "ppt"
    # Translation not applicable for PPT generation
    return await process_query(request, engine)

@app.get("/search", response_model=QueryResponse)
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    threshold: float = Query(0.35, description="Similarity threshold"),
    translate_to: str = Query('en', description="Target language for translation"),
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Search documents via GET request with translation"""
    return await process_query_get(q, "search", threshold, translate_to, engine)

@app.get("/explore", response_model=QueryResponse)
async def explore_documents_get(
    q: str = Query(..., description="Search query"),
    threshold: float = Query(0.35, description="Similarity threshold"),
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Explore documents via GET request (NO translation)"""
    return await process_query_get(q, "explore", threshold, 'en', engine)

@app.get("/think", response_model=QueryResponse)
async def think_documents_get(
    q: str = Query(..., description="Search query"),
    threshold: float = Query(0.35, description="Similarity threshold"),
    translate_to: str = Query('en', description="Target language for translation"),
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Think about documents via GET request with translation"""
    return await process_query_get(q, "think", threshold, translate_to, engine)

@app.get("/ppt", response_model=QueryResponse)
async def generate_ppt_get(
    q: str = Query(..., description="Search query"),
    threshold: float = Query(0.35, description="Similarity threshold"),
    engine: DocumentSearchEngine = Depends(get_engine)
):
    """Generate PPT via GET request (returns metadata incl. file path)."""
    return await process_query_get(q, "ppt", threshold, 'en', engine)

@app.get("/ppt/download")
async def download_ppt(path: str = Query(..., description="Absolute path returned by /ppt")):
    """Download a previously generated PPTX by providing its absolute path."""
    try:
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="File not found")
        filename = os.path.basename(path)
        return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation", filename=filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

@app.delete("/reset")
async def reset_system():
    """Reset the search engine (useful for development/testing)"""
    try:
        reset_search_engine()
        return {
            "success": True,
            "message": "Search engine reset successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reset system: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset system")

# ---------------------------
# Error Handlers
# ---------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# ---------------------------
# Startup/Shutdown Events
# ---------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("🚀 Starting Smart Document Search API with Translation...")
    try:
        # Initialize search engine
        engine = get_search_engine()
        health = engine.health_check()
        
        if health['status'] == 'healthy':
            logger.info(f"✅ Search engine initialized with {health['document_count']} documents")
        else:
            logger.warning(f"⚠️ Search engine status: {health['status']}")
            
        # Test translation service
        try:
            languages = get_supported_languages()
            logger.info(f"✅ Translation service initialized with {len(languages)} languages")
        except Exception as e:
            logger.warning(f"⚠️ Translation service error: {e}")
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise exception to allow API to start even with issues

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Smart Document Search API...")

# ---------------------------
# Additional Utility Endpoints
# ---------------------------
@app.get("/status")
async def get_status(engine: DocumentSearchEngine = Depends(get_engine)):
    """Get comprehensive system status"""
    try:
        health = engine.health_check()
        doc_info = engine.get_documents_info()
        
        # Test translation service
        try:
            translation_status = {
                "available": True,
                "supported_languages": len(get_supported_languages())
            }
        except Exception as e:
            translation_status = {
                "available": False,
                "error": str(e)
            }
        
        return {
            "api_status": "running",
            "search_engine_status": health['status'],
            "total_documents": doc_info['total_documents'],
            "clusters": doc_info['clusters'],
            "has_embeddings": health['has_embeddings'],
            "translation_service": translation_status,
            "timestamp": datetime.now().isoformat(),
            "uptime": "Available in production version"
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "api_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ---------------------------
# Run the application
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 8080))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info"
    )