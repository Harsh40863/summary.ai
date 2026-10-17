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

import sys

_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)
_summary_ai_dir = os.path.join(_project_root, "summary.ai")
for _p in [_backend_dir, _summary_ai_dir, _project_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")


from langchain_core.embeddings import Embeddings


def get_chunk_splitter():
    """Builds a token-based RecursiveCharacterTextSplitter with tiktoken encoding."""
    chunk_tokens = int(os.getenv("CHUNK_SIZE_TOKENS", "60"))
    overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "20"))
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_tokens,
            chunk_overlap=overlap_tokens,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
    except Exception:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_tokens * 4,
            chunk_overlap=overlap_tokens * 4,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )


class LocalEmbeddingAdapter(Embeddings):
    """LangChain embedding adapter backed by the app's existing embedder."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return get_embeddings(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return get_embeddings([text])[0].tolist()

class DocumentSearchEngine:
    """
    Core logic class for document search and analysis functionality
    """
    
    def __init__(self, user_email: Optional[str] = None):
        self.user_email = user_email
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
        try:
            self.documents = fetch_documents_from_db(self.user_email)
        except Exception as e:
            print(f"Warning: Failed to fetch documents from database: {e}")
            self.documents = []
        
        if not self.documents:
            print("No documents found in database. System will be ready after first upload.")
            # Initialize empty state instead of raising error
            self.doc_texts = []
            self.doc_embeddings = None
            self.index_to_docinfo = {}
            self.topic_labels = None
            self.search_engine = None
            self.n_clusters = 0
            return
        
        try:
            # Process documents
            valid_docs = []
            self.doc_texts = []
            embeddings = []
            texts_to_embed = []
            indices_to_embed = []
            
            for doc in self.documents:
                text = doc.get("content", "")
                if text.strip():
                    cleaned = clean_text(text)
                    valid_docs.append(doc)
                    self.doc_texts.append(cleaned)
                    
                    if doc.get("embedding"):
                        embeddings.append(doc["embedding"])
                    else:
                        embeddings.append(None)
                        texts_to_embed.append(cleaned)
                        indices_to_embed.append(len(embeddings) - 1)
            
            # Update self.documents to only include valid docs to stay in sync
            self.documents = valid_docs
            
            if not self.doc_texts:
                print("No valid text content found in documents.")
                self.doc_embeddings = None
                self.search_engine = None
                return
                
            if texts_to_embed:
                new_embs = get_embeddings(texts_to_embed)
                for i, idx in enumerate(indices_to_embed):
                    embeddings[idx] = new_embs[i]
                    
            self.doc_embeddings = np.array(embeddings)
            self.index_to_docinfo = {i: doc for i, doc in enumerate(self.documents)}
            
            # Create clusters
            self.n_clusters = min(len(self.documents), 3)
            if len(self.doc_embeddings) > 0:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(self.doc_embeddings)
                self.topic_labels = kmeans.labels_
            else:
                self.topic_labels = []
            
            # Initialize search engine
            if self.doc_embeddings is not None and len(self.doc_embeddings) > 0:
                self.search_engine = VectorSearchEngine(dimension=self.doc_embeddings.shape[1])
                self.search_engine.build_index(self.doc_embeddings, self.doc_texts)
            else:
                self.search_engine = None
                
        except Exception as e:
            print(f"Error processing documents: {e}")
            # Initialize empty state on error
            self.doc_texts = []
            self.doc_embeddings = None
            self.index_to_docinfo = {}
            self.topic_labels = None
            self.search_engine = None
            self.n_clusters = 0
    
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
                result = upload_file_to_db(uploaded_file, self.user_email)
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
        try:
            self.reload_documents()
        except Exception as e:
            print(f"Warning: Failed to reload documents after upload: {e}")
            # Don't fail the upload if reload fails
            
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

    def run_xpoz_agent(self, query: str, context: str = "") -> str:
        try:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                return "<div class='social-post' style='border: 1px solid #FFCC00;'><div class='post-header'>⚠️ Tavily Configuration Missing</div><div class='post-body'>Please add <b>TAVILY_API_KEY</b> to your backend <code>.env</code> file.</div></div>"
            
            mistral_api_key = os.getenv("MISTRAL_API_KEY")
            if not mistral_api_key:
                return "<div class='social-post' style='border: 1px solid #FFCC00;'><div class='post-header'>⚠️ Mistral Configuration Missing</div><div class='post-body'>Please add <b>MISTRAL_API_KEY</b> to your backend <code>.env</code> file.</div></div>"

            from langchain_mistralai import ChatMistralAI
            from langgraph.prebuilt import create_react_agent
            from langchain_core.messages import SystemMessage
            from langchain_core.tools import tool
            from tavily import TavilyClient
            
            model_name = os.getenv("MISTRAL_MODEL", "mistral-medium-3-5")
            llm = ChatMistralAI(model=model_name, api_key=mistral_api_key, temperature=0.1)
            
            @tool
            def web_search(search_query: str) -> str:
                """Search the web for recent and reliable information."""
                tavily = TavilyClient(api_key=tavily_api_key)
                results = tavily.search(query=search_query, max_results=5)
                out = []
                for r in results.get('results', []):
                    out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content']}\n")
                return "\n----\n".join(out)

            tools = [web_search]
            
            prompt_str = """You are a specialized research assistant agent. Your goal is to search the web for real information relevant to the user's query and the provided context.
            
            You must output your final answer EXACTLY in this HTML format:
            
            ## 🔍 Web Exploration for '[Your Extracted Search Term]'
            
            <div class='social-scroll-container'>
              <div class='social-post' style='border: 1px solid #00D2FF;'><div class='post-header'>🌐 [Domain Name / Source] - [Title]</div><div class='post-body'>[Content summary / Snippet]</div><a class='post-link' href='[url]' target='_blank'>Read Article</a></div>
              ...
            </div>
            
            Rules:
            1. Use the web_search tool to find the most relevant web pages based on the user's query.
            2. For each search result you select, create a corresponding `social-post` div inside the `social-scroll-container`.
            3. Make sure the HTML format is exactly as specified. Do not wrap it in markdown code blocks.
            4. Keep the HTML clean, and always provide direct links to the articles.
            5. If no information is found at all, output: ## 🔍 Web Exploration\n\n*No relevant information found.*
            """
            
            agent = create_react_agent(llm, tools, prompt=SystemMessage(content=prompt_str))
            
            input_message = f"Context: {context[:500]}\n\nQuery: {query}"
            result = agent.invoke({"messages": [("user", input_message)]})
            
            # The output is in the last message content
            output = result["messages"][-1].content
            
            # Clean up potential markdown formatting from LLM
            if output.startswith("```html"):
                output = output.replace("```html", "", 1)
            if output.endswith("```"):
                output = output[::-1].replace("```", "", 1)[::-1]
                
            return output.strip()
            
        except BaseException as e:
            import traceback
            traceback.print_exc()
            return f"<div class='social-post' style='border: 1px solid #FF9900;'><div class='post-header'>⚠️ Exploration Unavailable</div><div class='post-body'>Could not fetch web context at this time (Service error: {str(e)}).</div></div>"
    
    def _build_document_chunks(self) -> List[Dict[str, Any]]:
        splitter = get_chunk_splitter()
        chunks = []
        for doc_index, doc in enumerate(self.documents):
            content = doc.get("content", "")
            if not content.strip():
                continue
            split_chunks = splitter.split_text(content)
            for chunk_num, raw_chunk in enumerate(split_chunks):
                chunk_text = raw_chunk.lstrip(". ").strip()
                if not chunk_text:
                    continue
                sentences = sent_tokenize(chunk_text)
                first_sent = sentences[0] if sentences else ""
                chunks.append({
                    "text": chunk_text,
                    "doc_index": doc_index,
                    "chunk_index": chunk_num,
                    "best_sentence": first_sent,
                })
        return chunks

    def _mmr_chunk_search(self, query: str, threshold: float) -> List[Dict[str, Any]]:
        try:
            from langchain_core.documents import Document
            try:
                from langchain_community.vectorstores import FAISS
            except ImportError:
                from langchain.vectorstores import FAISS
        except ImportError:
            return []

        chunks = self._build_document_chunks()
        if not chunks:
            return []

        docs = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "doc_index": chunk["doc_index"],
                    "chunk_index": chunk["chunk_index"],
                    "best_sentence": chunk["best_sentence"],
                },
            )
            for chunk in chunks
        ]
        vectorstore = FAISS.from_documents(docs, LocalEmbeddingAdapter())
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": min(5, len(docs)),
                "fetch_k": min(20, len(docs)),
                "lambda_mult": 0.5,
            },
        )
        selected_docs = retriever.invoke(query)
        query_embedding = get_embeddings([clean_text(query)])
        results = []
        candidates = []

        for langchain_doc in selected_docs:
            doc_index = int(langchain_doc.metadata["doc_index"])
            doc = self.documents[doc_index]
            chunk = langchain_doc.page_content
            chunk_embedding = get_embeddings([chunk])
            score = float(np.dot(
                VectorSearchEngine._normalize(query_embedding),
                VectorSearchEngine._normalize(chunk_embedding).T,
            ).flatten()[0])
            entry = {
                "doc": doc,
                "best_sentence": langchain_doc.metadata.get("best_sentence", ""),
                "score": score,
                "relevant_chunk": chunk,
                "summary": generate_answer_with_rag(query, chunk, doc["name"]),
                "doc_index": doc_index,
            }
            candidates.append(entry)
            if score >= threshold:
                results.append(entry)

        if not results and candidates:
            best_candidate = max(candidates, key=lambda x: x["score"])
            if best_candidate["score"] > 0.1:
                results.append(best_candidate)

        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        if sorted_results and threshold > 0.20:
            top_score = sorted_results[0]["score"]
            sorted_results = [r for r in sorted_results if r["score"] >= top_score * 0.70]

        return sorted_results

    def find_relevant_documents(self, query: str, threshold: float = 0.35) -> List[Dict[str, Any]]:
        """
        Find documents relevant to the query
        
        Args:
            query: Search query string
            threshold: Similarity threshold for relevance
            
        Returns:
            List of relevant document results sorted by relevance score
        """
        if not self.documents or not self.doc_texts:
            return []

        mmr_results = self._mmr_chunk_search(query, threshold)
        if mmr_results:
            return mmr_results
            
        clean_query = clean_text(query)
        query_embedding = get_embeddings([clean_query])
        
        relevant_results = []
        candidates = []
        
        # 1. Fast document-level filtering via search engine index
        if self.search_engine is not None and self.search_engine.ntotal > 0:
            faiss_results = self.search_engine.search(query_embedding, top_k=2)
            top_doc_indices = [idx for idx, _score, _text in faiss_results]
        else:
            top_doc_indices = list(range(min(len(self.documents), 2)))
            
        # 2. Chunk-level search within top documents
        for idx in top_doc_indices:
            doc = self.documents[idx]
            content = doc.get("content", "")
            if not content.strip():
                continue

            splitter = get_chunk_splitter()
            raw_chunks = splitter.split_text(content)
            chunks = []
            chunk_first_sentence = []
            for raw_chunk in raw_chunks:
                chunk_text = raw_chunk.lstrip(". ").strip()
                if chunk_text:
                    chunks.append(chunk_text)
                    sents = sent_tokenize(chunk_text)
                    chunk_first_sentence.append(sents[0] if sents else "")

            if not chunks:
                continue

            chunks = chunks[:100]
            chunk_first_sentence = chunk_first_sentence[:100]

            chunk_embeddings = get_embeddings(chunks)
            # Use a temporary FAISS index for chunk-level search
            chunk_search = VectorSearchEngine.from_embeddings(chunk_embeddings, chunks)
            chunk_results = chunk_search.search(query_embedding, top_k=1)

            if not chunk_results:
                continue

            best_idx, best_score, relevant_chunk = chunk_results[0]
            best_sentence = chunk_first_sentence[best_idx] if best_idx < len(chunk_first_sentence) else ""

            candidate = {
                "doc": doc,
                "best_sentence": best_sentence,
                "score": float(best_score),
                "relevant_chunk": relevant_chunk,
                "doc_index": int(idx),
            }
            candidates.append(candidate)

            if best_score < threshold:
                continue
            
            # Generate query-specific summary
            query_summary = generate_answer_with_rag(query, relevant_chunk, doc["name"])
            
            relevant_results.append({
                "doc": doc,
                "best_sentence": best_sentence,
                "score": float(best_score),  # Convert to Python float
                "relevant_chunk": relevant_chunk,
                "summary": query_summary,
                "doc_index": int(idx)
            })
        
        # Sort by score (highest first)
        # Return only top 5 most relevant results
        sorted_results = sorted(relevant_results, key=lambda x: x["score"], reverse=True)
        if not sorted_results and candidates:
            fallback_results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]
            for result in fallback_results:
                result["summary"] = generate_answer_with_rag(query, result["relevant_chunk"], result["doc"]["name"])
            return fallback_results
        return sorted_results[:5]
    
    def search_action(self, query: str, threshold: float = 0.35) -> Dict[str, Any]:
        """
        Perform search action - uses LangChain MMR retriever on document chunks and returns relevant document chunks to the user.
        
        Args:
            query: Search query string
            threshold: Similarity threshold for relevance
            
        Returns:
            Dictionary with search results containing document chunks
        """
        relevant_results = self.find_relevant_documents(query, threshold)
        
        if not relevant_results:
            return {
                "success": False,
                "message": "No relevant document chunks found for your query.",
                "results": []
            }
            
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        llm = None
        if mistral_api_key:
            try:
                from langchain_core.messages import HumanMessage
                from langchain_mistralai import ChatMistralAI
                model_name = os.getenv("MISTRAL_MODEL", "mistral-medium-3-5")
                llm = ChatMistralAI(model=model_name, api_key=mistral_api_key, temperature=0.3)
            except Exception as e:
                print(f"Failed to initialize Mistral LLM for search: {e}")
        
        search_results = []
        for result in relevant_results:
            doc = result["doc"]
            chunk_text = result.get("relevant_chunk", result.get("summary", ""))
            
            summary = chunk_text
            if llm:
                try:
                    prompt = (
                        f"Based strictly on the following document excerpt, answer the query '{query}'.\n"
                        "RULES:\n"
                        "1. Rely ONLY on facts, figures, and details directly stated in the excerpt. Do not invent or extrapolate.\n"
                        "2. If the excerpt does not contain information to answer the question, state: 'Based on the provided documents, there is no relevant information available to answer this question.'\n"
                        "3. Answer thoroughly using clean markdown formatting (bolding, bullet points if helpful).\n\n"
                        f"Excerpt:\n{chunk_text}"
                    )
                    response = llm.invoke([HumanMessage(content=prompt)])
                    summary = response.content
                except Exception as e:
                    print(f"Mistral generation failed: {e}")
            
            search_results.append({
                "document_name": doc["name"],
                "best_sentence": result["best_sentence"],
                "score": result["score"],
                "summary": summary,
                "relevant_chunk": chunk_text,
                "action_type": "search"
            })
        
        return {
            "success": True,
            "message": f"Found {len(search_results)} relevant document chunks",
            "results": search_results,
            "action": "search"
        }
    
    def explore_action(self, query: str, threshold: float = 0.35) -> Dict[str, Any]:
        """
        Perform explore action - combines document insights with web search
        
        Args:
            query: Search query string
            threshold: Similarity threshold for relevance
            
        Returns:
            Dictionary with exploration results
        """
        relevant_results = self.find_relevant_documents(query, threshold)
        
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
            
            try:
                web_content = self.run_xpoz_agent(query, context=summary[:250])
                
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
    
    def think_action(self, query: str, threshold: float = 0.35) -> Dict[str, Any]:
        """
        Perform think action - generates refined insights using Google Gen AI
        
        Args:
            query: Search query string
            threshold: Similarity threshold for relevance
            
        Returns:
            Dictionary with thinking results
        """
        relevant_results = self.find_relevant_documents(query, threshold)
        
        if not relevant_results:
            return {
                "success": False,
                "message": "No relevant documents found for your query.",
                "results": []
            }
        
        think_results = []
        for result in relevant_results:
            doc = result["doc"]
            raw_chunk = result["relevant_chunk"]
            
            try:
                from think.pipeline import run_research_pipeline
                topic = f"{query}\n\nRelevant document context from {doc['name']}:\n{raw_chunk[:1500]}"
                state = run_research_pipeline(topic)
                refined_answer = (
                    f"{state.get('report', '')}\n\n"
                    f"Critic review:\n{state.get('feedback', '')}"
                ).strip()
                
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
                    "refined_insight": f"Error during think pipeline: {str(e)}",
                    "action_type": "think",
                    "error": True
                })
        
        return {
            "success": True,
            "message": f"Generated insights for {len(think_results)} relevant documents",
            "results": think_results,
            "action": "think"
        }
    
    def process_query(self, query: str, action: str, threshold: float = 0.35) -> Dict[str, Any]:
        """
        Main method to process query with specified action
        
        Args:
            query: Search query string
            action: Action type ("search", "explore", "think")
            threshold: Similarity threshold for document relevance
            
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
            return self.search_action(query, threshold)
        elif action == "explore":
            return self.explore_action(query, threshold)
        elif action == "think":
            return self.think_action(query, threshold)
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
                "id": doc.get("doc_id"),
                "name": doc["name"],
                "content_length": len(doc["content"]),
                "upload_date": str(doc.get("uploaded_at") or doc.get("upload_date", "Unknown")),
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
            has_docs = len(self.documents) > 0 if self.documents else False
            has_embeddings = self.doc_embeddings is not None
            has_search_engine = self.search_engine is not None
            
            # System is healthy if it has documents and embeddings, or if it's empty but ready
            status = "healthy" if (has_docs and has_embeddings and has_search_engine) or (not has_docs and not has_embeddings) else "unhealthy"
            
            return {
                "status": status,
                "has_documents": has_docs,
                "has_embeddings": has_embeddings,
                "has_search_engine": has_search_engine,
                "document_count": len(self.documents) if self.documents else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Per-user instances for reuse without leaking documents between users.
_search_engine_instances = {}

def get_search_engine(user_email: Optional[str] = None) -> DocumentSearchEngine:
    """
    Get or create global search engine instance
    
    Returns:
        DocumentSearchEngine instance
    """
    cache_key = user_email or "__anonymous__"
    if cache_key not in _search_engine_instances:
        _search_engine_instances[cache_key] = DocumentSearchEngine(user_email=user_email)
    return _search_engine_instances[cache_key]

def reset_search_engine(user_email: Optional[str] = None):
    """Reset global search engine instance (useful for testing or reinitialization)"""
    if user_email is None:
        _search_engine_instances.clear()
    else:
        _search_engine_instances.pop(user_email, None)
