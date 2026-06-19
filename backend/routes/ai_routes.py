import sys
import os

# Add summary.ai to path to import its modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../summary.ai')))

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from typing import List
from auth import get_current_user
from datetime import datetime

# Import summary.ai models and logic
from api import QueryRequest, QueryResponse, UploadResponse, SupportedLanguagesResponse, HealthResponse, DocumentsResponse
from backend.logic import get_search_engine, DocumentSearchEngine
from translation.redis_translator import get_supported_languages
from uploadmdb.db_ingest import delete_document_from_db

router = APIRouter()

def get_engine_for_user(current_email: str = Depends(get_current_user)) -> DocumentSearchEngine:
    try:
        engine = get_search_engine(current_email)
        return engine
    except Exception as e:
        raise HTTPException(status_code=500, detail="Search engine initialization failed")

@router.get("/health", response_model=HealthResponse)
def health_check(engine: DocumentSearchEngine = Depends(get_engine_for_user)):
    try:
        health_data = engine.health_check()
        return HealthResponse(**health_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/languages", response_model=SupportedLanguagesResponse)
def get_languages(current_email: str = Depends(get_current_user)):
    try:
        return SupportedLanguagesResponse(
            languages=get_supported_languages(),
            default="en",
            note="Translation available for search and think actions."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get supported languages")

@router.post("/upload", response_model=UploadResponse)
def upload_files(
    files: List[UploadFile] = File(...),
    engine: DocumentSearchEngine = Depends(get_engine_for_user)
):
    try:
        allowed_extensions = {'.pdf', '.docx', '.pptx', '.txt', '.jpg', '.jpeg', '.png'}
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="File must have a filename")
            file_ext = os.path.splitext(file.filename.lower())[1]
            if file_ext not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"File type not supported: {file.filename}")
        
        upload_results = engine.upload_files(files)
        success_count = sum(1 for result in upload_results if result.get('status') == 'success')
        return UploadResponse(
            success=success_count > 0,
            message=f"Successfully uploaded {success_count}/{len(files)} files",
            uploaded_files=upload_results,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@router.post("/query", response_model=QueryResponse)
def process_query(
    request: QueryRequest,
    engine: DocumentSearchEngine = Depends(get_engine_for_user)
):
    try:
        from translation.redis_translator import is_supported_language, translate_response
        valid_actions = ['search', 'explore', 'think', 'ppt']
        if request.action.lower() not in valid_actions:
            raise HTTPException(status_code=400, detail=f"Invalid action '{request.action}'")
            
        if request.translate_to and request.translate_to != 'en' and request.action.lower() in ['search', 'think']:
            if not is_supported_language(request.translate_to):
                raise HTTPException(status_code=400, detail=f"Unsupported language: {request.translate_to}")
                
        result = engine.process_query(request.query, request.action, request.threshold)
        
        if request.translate_to and request.translate_to != 'en' and request.action.lower() in ['search', 'think']:
            result = translate_response(result, request.translate_to)
            
        if not result['success']:
            from api import TranslationInfo
            return QueryResponse(
                success=False,
                message=result['message'],
                results=[],
                action=request.action,
                timestamp=datetime.now().isoformat(),
                query=request.query,
                translation=TranslationInfo(**result['translation']) if 'translation' in result else None
            )
            
        from api import TranslationInfo
        return QueryResponse(
            success=True,
            message=result['message'],
            results=result['results'],
            action=request.action,
            timestamp=datetime.now().isoformat(),
            query=request.query,
            translation=TranslationInfo(**result['translation']) if 'translation' in result else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.get("/documents", response_model=DocumentsResponse)
def get_documents_info(engine: DocumentSearchEngine = Depends(get_engine_for_user)):
    try:
        from api import DocumentInfo
        doc_info = engine.get_documents_info()
        return DocumentsResponse(
            total_documents=doc_info['total_documents'],
            documents=[DocumentInfo(**doc) for doc in doc_info['documents']],
            clusters=doc_info['clusters'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve documents information")

@router.post("/documents/reload")
def reload_documents(engine: DocumentSearchEngine = Depends(get_engine_for_user)):
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
        raise HTTPException(status_code=500, detail="Failed to reload documents")

@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, current_email: str = Depends(get_current_user), engine: DocumentSearchEngine = Depends(get_engine_for_user)):
    try:
        success = delete_document_from_db(doc_id, current_email)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or could not be deleted")
        
        # Reload the search engine's in-memory cache to reflect the deletion
        engine.reload_documents()
        return {"success": True, "message": "Document deleted successfully"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
