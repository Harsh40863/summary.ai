# ---------------------------
# Core Application Logic
# ---------------------------
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import requests
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
from sklearn.metrics.pairwise import cosine_similarity

# Custom modules
from text_preprocessing.cleaner import clean_text
from embedding.embedder import get_embeddings
from search.vector_search import VectorSearchEngine
from summarization.summarizer import flan_summarize_with_query as summarize_text_with_query
from rag.rag_engine import generate_answer_with_rag
from rag.langchain_tools import GoogleSearchTool
from uploadmdb.db_ingest import upload_file_to_db, fetch_documents_from_db

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

class DocumentSearchEngine:
    """
    Core logic class for document search and analysis functionality
    """
    
    def __init__(self):
        self.documents = []
        self.doc_texts = []
        self.doc_embeddings = None
        self.index_to_docinfo = {}
        self.topic_labels = None
        self.search_engine = None
        self.n_clusters = 3
        self._initialize_nltk()
        self._load_documents()
        
    def _initialize_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
    
    def _load_documents(self):
        """Load and preprocess documents from database"""
        self.documents = fetch_documents_from_db()
        
        if not self.documents:
            raise ValueError("No valid documents found in the database.")
        
        # Process documents
        self.doc_texts = [clean_text(doc["content"]) for doc in self.documents]
        self.doc_embeddings = get_embeddings(self.doc_texts)
        self.index_to_docinfo = {i: doc for i, doc in enumerate(self.documents)}
        
        # Create clusters
        self.n_clusters = min(len(self.documents), 3)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(self.doc_embeddings)
        self.topic_labels = kmeans.labels_
        
        # Initialize search engine
        self.search_engine = VectorSearchEngine(dimension=self.doc_embeddings.shape[1])
        self.search_engine.build_index(self.doc_embeddings, self.doc_texts)
    
    def reload_documents(self):
        """Reload documents from database (useful after new uploads)"""
        self._load_documents()
    
    def upload_files(self, uploaded_files: List[Any]) -> List[Dict[str, str]]:
        """
        Upload multiple files to database
        
        Args:
            uploaded_files: List of file objects
            
        Returns:
            List of upload results with filename and status
        """
        results = []
        for uploaded_file in uploaded_files:
            try:
                result = upload_file_to_db(uploaded_file)
                results.append({
                    "filename": uploaded_file.filename if hasattr(uploaded_file, 'filename') else str(uploaded_file),
                    "status": "success",
                    "message": result
                })
            except Exception as e:
                results.append({
                    "filename": uploaded_file.filename if hasattr(uploaded_file, 'filename') else str(uploaded_file),
                    "status": "error",
                    "message": str(e)
                })
        
        # Reload documents after upload
        self.reload_documents()
        return results
    
    def google_search(self, query: str, api_key: str = None, cse_id: str = None, num: int = 5) -> Dict[str, Any]:
        """
        Perform Google Custom Search
        
        Args:
            query: Search query string
            api_key: Google API key (uses env variable if not provided)
            cse_id: Custom Search Engine ID (uses env variable if not provided)
            num: Number of results to return
            
        Returns:
            Dictionary with search results or error
        """
        api_key = api_key or GOOGLE_API_KEY
        cse_id = cse_id or GOOGLE_CSE_ID
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"q": query, "key": api_key, "cx": cse_id, "num": num}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def find_relevant_documents(self, query: str, threshold: float = 0.80) -> List[Dict[str, Any]]:
        """
        Find documents relevant to the query
        
        Args:
            query: Search query string
            threshold: Similarity threshold for relevance
            
        Returns:
            List of relevant document results sorted by relevance score
        """
        clean_query = clean_text(query)
        query_embedding = get_embeddings([clean_query])
        
        relevant_results = []
        
        for idx, doc in enumerate(self.documents):
            sentences = sent_tokenize(doc["content"])
            if not sentences:
                continue
            
            sentence_embeddings = get_embeddings(sentences)
            # Use cosine similarity instead of dot product
            sim_scores = cosine_similarity(sentence_embeddings, query_embedding).flatten()
            best_score = np.max(sim_scores)
            best_idx = np.argmax(sim_scores)
            
            if best_score < threshold:
                continue
            
            # Get relevant context chunk
            start = max(0, best_idx - 1)
            end = min(len(sentences), best_idx + 2)
            relevant_chunk = " ".join(sentences[start:end])
            
            # Generate query-specific summary
            query_summary = generate_answer_with_rag(query, relevant_chunk, doc["name"])
            
            relevant_results.append({
                "doc": doc,
                "best_sentence": sentences[best_idx],
                "score": float(best_score),  # Convert to Python float
                "relevant_chunk": relevant_chunk,
                "summary": query_summary,
                "doc_index": idx
            })
        
        # Sort by score (highest first)
        # Return only top 5 most relevant results
            sorted_results = sorted(relevant_results, key=lambda x: x["score"], reverse=True)
            return sorted_results[:5]
    
    def search_action(self, query: str) -> Dict[str, Any]:
        """
        Perform search action - returns document summaries
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with search results and metadata
        """
        relevant_results = self.find_relevant_documents(query)
        
        if not relevant_results:
            return {
                "success": False,
                "message": "No relevant documents found for your query.",
                "results": []
            }
        
        search_results = []
        for result in relevant_results:
            doc = result["doc"]
            search_results.append({
                "document_name": doc["name"],
                "best_sentence": result["best_sentence"],
                "score": result["score"],
                "summary": result["summary"],
                "action_type": "search"
            })
        
        return {
            "success": True,
            "message": f"Found {len(search_results)} relevant documents",
            "results": search_results,
            "action": "search"
        }
    
    def explore_action(self, query: str) -> Dict[str, Any]:
        """
        Perform explore action - combines document insights with web search
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with exploration results
        """
        relevant_results = self.find_relevant_documents(query)
        
        if not relevant_results:
            return {
                "success": False,
                "message": "No relevant documents found for your query.",
                "results": []
            }
        
        explore_results = []
        for result in relevant_results:
            doc = result["doc"]
            summary = result["summary"]
            
            # Enhanced query for web search
            enhanced_query = f"{query} - Context: {summary[:250]}"
            
            try:
                search_tool = GoogleSearchTool(api_key=GOOGLE_API_KEY2, cse_id=GOOGLE_CSE_ID)
                web_content = search_tool.run(enhanced_query)
                
                explore_results.append({
                    "document_name": doc["name"],
                    "best_sentence": result["best_sentence"],
                    "score": result["score"],
                    "web_content": web_content,
                    "action_type": "explore"
                })
            except Exception as e:
                explore_results.append({
                    "document_name": doc["name"],
                    "best_sentence": result["best_sentence"],
                    "score": result["score"],
                    "web_content": f"Error fetching web content: {str(e)}",
                    "action_type": "explore",
                    "error": True
                })
        
        return {
            "success": True,
            "message": f"Explored {len(explore_results)} relevant documents",
            "results": explore_results,
            "action": "explore"
        }
    
    def think_action(self, query: str) -> Dict[str, Any]:
        """
        Perform think action - generates refined insights using Google Gen AI
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with thinking results
        """
        relevant_results = self.find_relevant_documents(query)
        
        if not relevant_results:
            return {
                "success": False,
                "message": "No relevant documents found for your query.",
                "results": []
            }
        
        think_results = []
        for result in relevant_results:
            doc = result["doc"]
            summary = result["summary"]
            
            try:
                from rag.Google_Gen_AI import generate_answer_with_google
                refined_answer = generate_answer_with_google(query, summary, doc["name"])
                
                think_results.append({
                    "document_name": doc["name"],
                    "best_sentence": result["best_sentence"],
                    "score": result["score"],
                    "refined_insight": refined_answer,
                    "action_type": "think"
                })
            except Exception as e:
                think_results.append({
                    "document_name": doc["name"],
                    "best_sentence": result["best_sentence"],
                    "score": result["score"],
                    "refined_insight": f"Error during insight generation: {str(e)}",
                    "action_type": "think",
                    "error": True
                })
        
        return {
            "success": True,
            "message": f"Generated insights for {len(think_results)} relevant documents",
            "results": think_results,
            "action": "think"
        }
    
    def process_query(self, query: str, action: str) -> Dict[str, Any]:
        """
        Main method to process query with specified action
        
        Args:
            query: Search query string
            action: Action type ("search", "explore", "think")
            
        Returns:
            Dictionary with results based on action type
        """
        if not query or not query.strip():
            return {
                "success": False,
                "message": "Query cannot be empty",
                "results": []
            }
        
        action = action.lower().strip()
        
        if action == "search":
            return self.search_action(query)
        elif action == "explore":
            return self.explore_action(query)
        elif action == "think":
            return self.think_action(query)
        else:
            return {
                "success": False,
                "message": f"Unknown action: {action}. Available actions: search, explore, think",
                "results": []
            }
    
    def get_documents_info(self) -> Dict[str, Any]:
        """
        Get information about loaded documents
        
        Returns:
            Dictionary with document statistics and info
        """
        if not self.documents:
            return {
                "total_documents": 0,
                "documents": [],
                "clusters": 0
            }
        
        doc_info = []
        for doc in self.documents:
            doc_info.append({
                "name": doc["name"],
                "content_length": len(doc["content"]),
                "upload_date": doc.get("upload_date", "Unknown")
            })
        
        return {
            "total_documents": len(self.documents),
            "documents": doc_info,
            "clusters": self.n_clusters
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the system is ready to process queries
        
        Returns:
            Dictionary with system health status
        """
        try:
            has_docs = len(self.documents) > 0
            has_embeddings = self.doc_embeddings is not None
            has_search_engine = self.search_engine is not None
            
            return {
                "status": "healthy" if all([has_docs, has_embeddings, has_search_engine]) else "unhealthy",
                "has_documents": has_docs,
                "has_embeddings": has_embeddings,
                "has_search_engine": has_search_engine,
                "document_count": len(self.documents),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global instance for reuse
_search_engine_instance = None

def get_search_engine() -> DocumentSearchEngine:
    """
    Get or create global search engine instance
    
    Returns:
        DocumentSearchEngine instance
    """
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = DocumentSearchEngine()
    return _search_engine_instance

def reset_search_engine():
    """Reset global search engine instance (useful for testing or reinitialization)"""
    global _search_engine_instance
    _search_engine_instance = None