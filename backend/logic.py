# ---------------------------
# Core Application Logic - IMPROVED VERSION WITH FIXED RELEVANCE
# ---------------------------
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import requests
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from pptx import Presentation
import re

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
    Core logic class for document search and analysis functionality - IMPROVED VERSION
    """
    
    def __init__(self):
        self.documents = []
        self.doc_texts = []
        self.doc_embeddings = None
        self.index_to_docinfo = {}
        self.topic_labels = None
        self.search_engine = None
        self.n_clusters = 3
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._initialize_nltk()
        self._load_documents()
        
    def _initialize_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")
    
    def _load_documents(self):
        """Load and preprocess documents from database"""
        self.documents = fetch_documents_from_db()
        
        if not self.documents:
            raise ValueError("No valid documents found in the database.")
        
        # Process documents
        self.doc_texts = [clean_text(doc["content"]) for doc in self.documents]
        self.doc_embeddings = get_embeddings(self.doc_texts)
        self.index_to_docinfo = {i: doc for i, doc in enumerate(self.documents)}
        
        # Create TF-IDF vectorizer for keyword matching
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_texts)
        
        # Create clusters
        self.n_clusters = min(len(self.documents), 3)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(self.doc_embeddings)
        self.topic_labels = kmeans.labels_
        
        # Initialize search engine
        self.search_engine = VectorSearchEngine(dimension=self.doc_embeddings.shape[1])
        self.search_engine.build_index(self.doc_embeddings, self.doc_texts)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text with better filtering"""
        words = word_tokenize(text.lower())
        # Comprehensive stopwords list
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'its', 'our', 'their', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'also', 'from', 'up',
            'about', 'into', 'over', 'after', 'as', 'because', 'if', 'then', 'there', 'here'
        }
        
        # Filter keywords with stricter criteria
        keywords = [
            word for word in words 
            if (len(word) > 3 and  # Minimum 4 characters
                word not in stopwords and 
                word.isalpha() and  # Only alphabetic characters
                not word.isdigit())  # Not numbers
        ]
        return list(set(keywords))
    
    def _calculate_embedding_similarity(self, query: str, document_text: str) -> float:
        """Calculate embedding similarity between query and document"""
        try:
            query_embedding = get_embeddings([clean_text(query)])
            doc_embedding = get_embeddings([clean_text(document_text)])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    def _check_keyword_overlap(self, query: str, document_text: str) -> Tuple[float, int]:
        """Enhanced keyword overlap check with more details"""
        query_keywords = set(self._extract_keywords(query))
        doc_keywords = set(self._extract_keywords(document_text))
        
        if not query_keywords:
            return 0.0, 0
        
        # Calculate intersection
        intersection = query_keywords.intersection(doc_keywords)
        union = query_keywords.union(doc_keywords)
        
        # Jaccard similarity
        jaccard_score = len(intersection) / len(union) if union else 0.0
        
        # Calculate coverage of query keywords in document
        query_coverage = len(intersection) / len(query_keywords) if query_keywords else 0.0
        
        # Check for exact phrase matches (case-insensitive)
        query_lower = query.lower()
        doc_lower = document_text.lower()
        
        # Look for multi-word phrases from query in document
        phrase_bonus = 0.0
        query_words = query.split()
        if len(query_words) > 1:
            # Check for consecutive word matches
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase.lower() in doc_lower:
                    phrase_bonus += 0.2
        
        # Single word exact matches
        if query_lower in doc_lower:
            phrase_bonus += 0.1
        
        # Combine scores with emphasis on query coverage
        final_score = (jaccard_score * 0.3) + (query_coverage * 0.5) + (phrase_bonus * 0.2)
        
        return final_score, len(intersection)
    
    def _semantic_relevance_check(self, query: str, document_text: str, strict_threshold: float = 0.25) -> Tuple[bool, Dict[str, float]]:
        """
        Enhanced multi-signal relevance check with RELAXED thresholds for better recall
        """
        scores = {}
        
        # 1. Embedding similarity
        embedding_score = self._calculate_embedding_similarity(query, document_text)
        scores['embedding'] = embedding_score
        
        # 2. Keyword overlap analysis with improved matching
        keyword_score, keyword_matches = self._check_keyword_overlap(query, document_text)
        scores['keyword'] = keyword_score
        scores['keyword_matches'] = keyword_matches
        
        # 3. Enhanced direct string matching (case-insensitive)
        query_lower = query.lower()
        doc_lower = document_text.lower()
        
        # Check for exact phrase matches
        phrase_matches = 0
        exact_match_score = 0.0
        
        # Split query into meaningful terms
        query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
        
        for term in query_terms:
            if term in doc_lower:
                exact_match_score += 1.0
                # Check for context around the term
                if f" {term} " in doc_lower or doc_lower.startswith(term) or doc_lower.endswith(term):
                    phrase_matches += 1
        
        # Normalize by number of query terms
        if query_terms:
            exact_match_score = exact_match_score / len(query_terms)
            phrase_match_score = phrase_matches / len(query_terms)
        else:
            exact_match_score = 0.0
            phrase_match_score = 0.0
        
        scores['exact_match'] = exact_match_score
        scores['phrase_match'] = phrase_match_score
        
        # 4. TF-IDF similarity (with error handling)
        try:
            query_tfidf = self.tfidf_vectorizer.transform([clean_text(query)])
            doc_idx = next((i for i, text in enumerate(self.doc_texts) if text == clean_text(document_text)), -1)
            
            if doc_idx >= 0:
                tfidf_score = float((query_tfidf * self.tfidf_matrix[doc_idx].T).toarray()[0, 0])
            else:
                doc_tfidf = self.tfidf_vectorizer.transform([clean_text(document_text)])
                tfidf_score = float((query_tfidf * doc_tfidf.T).toarray()[0, 0])
        except Exception as e:
            print(f"TF-IDF calculation error: {e}")
            tfidf_score = 0.0
        scores['tfidf'] = tfidf_score
        
        # 5. Topic/Entity matching (for cases like "who is harry potter" -> "harry potter synopsis")
        # Extract potential entities from query
        query_entities = []
        if query.lower().startswith(('who is', 'what is', 'tell me about')):
            # Extract the main subject
            entity_part = query.lower()
            for prefix in ['who is ', 'what is ', 'tell me about ']:
                if entity_part.startswith(prefix):
                    entity_part = entity_part[len(prefix):].strip()
                    break
            query_entities.append(entity_part)
        else:
            # Use all meaningful terms as potential entities
            query_entities = [term for term in query.split() if len(term) > 2]
        
        # Check if any query entities appear in document title or content
        entity_match_score = 0.0
        doc_title = document_text[:200].lower()  # Check first 200 chars for title-like content
        
        for entity in query_entities:
            entity_lower = entity.lower()
            if entity_lower in doc_lower:
                entity_match_score += 0.5
                # Bonus if entity appears early in document (likely in title/summary)
                if entity_lower in doc_title:
                    entity_match_score += 0.3
        
        scores['entity_match'] = min(entity_match_score, 1.0)
        
        # Calculate weighted final score with RELAXED weighting
        final_score = (
            embedding_score * 0.25 +           # Reduced weight for embedding
            keyword_score * 0.20 +             # Keyword relevance
            exact_match_score * 0.25 +         # Direct matches (important)
            phrase_match_score * 0.15 +        # Phrase context
            entity_match_score * 0.15          # Entity/topic matching
        )
        
        scores['final'] = final_score
        
        # RELAXED threshold checks for better recall
        is_relevant = (
            final_score >= strict_threshold or  # Lower base threshold
            (embedding_score >= 0.20 and exact_match_score >= 0.5) or  # Good semantic + exact match
            (entity_match_score >= 0.5 and exact_match_score >= 0.3) or  # Good entity match
            (exact_match_score >= 0.7) or      # Strong exact matches alone
            (keyword_matches >= 2 and embedding_score >= 0.15)  # Multiple keywords + some semantic similarity
        )
        
        return is_relevant, scores
    
    def find_relevant_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents relevant to the query with IMPROVED relevance checking
        """
        if not query or len(query.strip()) < 2:
            return []
        
        clean_query = clean_text(query)
        if not clean_query:
            clean_query = query.strip()  # Fallback to original query if cleaning removes everything
        
        query_embedding = get_embeddings([clean_query])
        relevant_results = []
        
        # More lenient pre-filtering
        query_words = set(word.lower() for word in query.split() if len(word) > 1)  # Reduced from 2 to 1
        
        for idx, doc in enumerate(self.documents):
            doc_content = doc["content"]
            doc_name = doc.get("name", "").lower()
            
            # Enhanced pre-filter: check document name AND content
            doc_words_content = set(word.lower() for word in doc_content[:1000].split())  # Check first 1000 chars
            doc_words_name = set(word.lower() for word in doc_name.split())
            all_doc_words = doc_words_content.union(doc_words_name)
            
            # More lenient pre-filtering - allow if ANY query word matches
            if not query_words.intersection(all_doc_words) and len(query_words) > 0:
                # Additional check for partial matches
                has_partial_match = False
                for query_word in query_words:
                    for doc_word in all_doc_words:
                        if query_word in doc_word or doc_word in query_word:
                            has_partial_match = True
                            break
                    if has_partial_match:
                        break
                
                if not has_partial_match:
                    continue
            
            # Full semantic relevance check with relaxed threshold
            is_relevant, scores = self._semantic_relevance_check(query, doc_content, strict_threshold=0.15)
            
            if not is_relevant:
                continue
            
            # Sentence-level analysis
            sentences = sent_tokenize(doc_content)
            if not sentences:
                # If sentence tokenization fails, use the whole document
                sentences = [doc_content[:500]]  # First 500 chars as fallback
            
            try:
                sentence_embeddings = get_embeddings(sentences[:10])  # Limit to first 10 sentences for performance
                sim_scores = np.dot(sentence_embeddings, query_embedding.T).flatten()
                
                # More lenient sentence filtering
                good_sentence_indices = np.where(sim_scores >= 0.1)[0]  # Reduced from 0.2 to 0.1
                if len(good_sentence_indices) == 0:
                    # If no sentences meet threshold, use the best one anyway
                    best_sentence_idx = np.argmax(sim_scores)
                    good_sentence_indices = [best_sentence_idx]
                
                best_sentence_idx = good_sentence_indices[np.argmax(sim_scores[good_sentence_indices])]
                best_sentence_score = sim_scores[best_sentence_idx]
                
                # Create relevant chunk
                top_indices = good_sentence_indices[np.argsort(sim_scores[good_sentence_indices])[-3:]]
                relevant_sentences = [sentences[i] for i in top_indices]
                relevant_chunk = " ".join(relevant_sentences)
                
            except Exception as e:
                print(f"Error in sentence analysis: {e}")
                # Fallback to first part of document
                relevant_chunk = doc_content[:500]
                best_sentence_score = scores['embedding']
                best_sentence_idx = 0
            
            # Generate RAG answer with error handling
            try:
                rag_answer = generate_answer_with_rag(query, relevant_chunk, doc["name"])
                
                # More lenient RAG answer filtering
                no_info_phrases = [
                    "not mentioned", "not found", "no information", "does not contain",
                    "not discussed", "not provided", "cannot find", "not specified"
                ]
                
                answer_lower = rag_answer.lower()
                # Only skip if RAG explicitly and clearly says no relevant info
                definitive_no_info = any(
                    phrase in answer_lower and 
                    ("clearly" in answer_lower or "definitely" in answer_lower or "explicitly" in answer_lower)
                    for phrase in no_info_phrases
                )
                
                if definitive_no_info:
                    continue
                    
                # Accept shorter answers if they contain relevant keywords
                if len(rag_answer.strip()) < 10:
                    continue
                    
            except Exception as e:
                print(f"RAG generation error: {e}")
                rag_answer = f"Document contains relevant information about: {query}"
            
            # Calculate comprehensive relevance score
            # Normalize the embedding score (it seems to be scaled differently)
            normalized_embedding = min(best_sentence_score / 100.0, 1.0) if best_sentence_score > 10 else best_sentence_score
            
            final_relevance_score = (
                scores['final'] * 0.6 +              # Increased weight for semantic final score
                normalized_embedding * 0.2 +         # Normalized embedding contribution
                scores.get('entity_match', 0) * 0.2  # Entity matching bonus
            )
            
            try:
                best_sentence = sentences[best_sentence_idx] if sentences else relevant_chunk[:200]
            except:
                best_sentence = relevant_chunk[:200]
            
            relevant_results.append({
                "doc": doc,
                "best_sentence": best_sentence,
                "score": float(final_relevance_score),
                "embedding_score": float(best_sentence_score),
                "semantic_scores": scores,
                "relevant_chunk": relevant_chunk,
                "summary": rag_answer,
                "doc_index": idx,
                "keyword_matches": scores.get('keyword_matches', 0)
            })
        
        # Sort by relevance score
        sorted_results = sorted(relevant_results, key=lambda x: x["score"], reverse=True)
        
        # MUCH MORE STRICT quality filter to eliminate irrelevant results
        high_quality_results = [
            result for result in sorted_results 
            if (result["score"] >= 0.4 and  # Higher threshold
                result["semantic_scores"]["final"] >= 0.35 and  # Good overall semantic score
                (result["semantic_scores"]["exact_match"] >= 0.5 or  # Good exact matches OR
                 result["semantic_scores"].get("entity_match", 0) >= 0.5 or  # Good entity matches OR
                 result["keyword_matches"] >= 2))  # Multiple keyword matches
        ]
        
        # If still no results, take the best ones anyway
        if not high_quality_results and sorted_results:
            high_quality_results = sorted_results[:2]  # Take top 2 regardless of score
        
        return high_quality_results[:max_results]
    
    def reload_documents(self):
        """Reload documents from database (useful after new uploads)"""
        self._load_documents()
    
    def upload_files(self, uploaded_files: List[Any]) -> List[Dict[str, str]]:
        """Upload multiple files to database"""
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
    
    def search_action(self, query: str) -> Dict[str, Any]:
        """Perform search action with IMPROVED error handling"""
        if not query or len(query.strip()) < 2:
            return {
                "success": False,
                "message": "Query must be at least 2 characters long",
                "results": []
            }
        
        relevant_results = self.find_relevant_documents(query, max_results=5)
        
        if not relevant_results:
            # Provide more helpful feedback
            doc_info = self.get_documents_info()
            available_docs = [doc["name"] for doc in doc_info["documents"]]
            
            return {
                "success": False,
                "message": f"No documents found containing information about '{query}'.\n\nAvailable documents: {', '.join(available_docs) if available_docs else 'None'}\n\nTips:\n- Try using different keywords\n- Check document content matches your query\n- Use simpler terms",
                "results": [],
                "available_documents": available_docs
            }
        
        search_results = []
        for result in relevant_results:
            doc = result["doc"]
            search_results.append({
                "document_name": doc["name"],
                "best_sentence": result["best_sentence"],
                "relevance_score": result["score"],
                "embedding_score": result["embedding_score"],
                "keyword_matches": result["keyword_matches"],
                "summary": result["summary"],
                "action_type": "search",
                "confidence": "high" if result["score"] >= 0.3 else ("medium" if result["score"] >= 0.2 else "low"),
                "semantic_scores": result["semantic_scores"]  # For debugging
            })
        
        return {
            "success": True,
            "message": f"Found {len(search_results)} relevant documents for '{query}'",
            "results": search_results,
            "action": "search",
            "query_processed": query
        }
    
    def explore_action(self, query: str) -> Dict[str, Any]:
        """Perform explore action - combines document insights with web search"""
        relevant_results = self.find_relevant_documents(query, max_results=3)
        
        if not relevant_results:
            return {
                "success": False,
                "message": f"No documents found containing relevant information about '{query}' for exploration. Please ensure you have uploaded relevant documents first.",
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
                    "relevance_score": result["score"],
                    "keyword_matches": result["keyword_matches"],
                    "web_content": web_content,
                    "action_type": "explore"
                })
            except Exception as e:
                explore_results.append({
                    "document_name": doc["name"],
                    "best_sentence": result["best_sentence"],
                    "relevance_score": result["score"],
                    "keyword_matches": result["keyword_matches"],
                    "web_content": f"Error fetching web content: {str(e)}",
                    "action_type": "explore",
                    "error": True
                })
        
        return {
            "success": True,
            "message": f"Explored {len(explore_results)} relevant documents for '{query}'",
            "results": explore_results,
            "action": "explore"
        }
    
    def think_action(self, query: str) -> Dict[str, Any]:
        """Perform think action - generates refined insights using Google Gen AI"""
        relevant_results = self.find_relevant_documents(query, max_results=3)
        
        if not relevant_results:
            return {
                "success": False,
                "message": f"No documents found containing relevant information about '{query}' for generating insights. Please ensure you have uploaded relevant documents first.",
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
                    "relevance_score": result["score"],
                    "keyword_matches": result["keyword_matches"],
                    "refined_insight": refined_answer,
                    "action_type": "think"
                })
            except Exception as e:
                think_results.append({
                    "document_name": doc["name"],
                    "best_sentence": result["best_sentence"],
                    "relevance_score": result["score"],
                    "keyword_matches": result["keyword_matches"],
                    "refined_insight": f"Error during insight generation: {str(e)}",
                    "action_type": "think",
                    "error": True
                })
        
        return {
            "success": True,
            "message": f"Generated insights for {len(think_results)} relevant documents about '{query}'",
            "results": think_results,
            "action": "think"
        }
    
    def process_query(self, query: str, action: str) -> Dict[str, Any]:
        """Main method to process query with specified action"""
        if not query or not query.strip():
            return {
                "success": False,
                "message": "Query cannot be empty",
                "results": []
            }
        
        # Clean and validate query
        query = query.strip()
        if len(query) < 2:
            return {
                "success": False,
                "message": "Query must be at least 2 characters long",
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
        """Get information about loaded documents"""
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
        """Check if the system is ready to process queries"""
        try:
            has_docs = len(self.documents) > 0
            has_embeddings = self.doc_embeddings is not None
            has_search_engine = self.search_engine is not None
            has_tfidf = self.tfidf_vectorizer is not None
            
            return {
                "status": "healthy" if all([has_docs, has_embeddings, has_search_engine, has_tfidf]) else "unhealthy",
                "has_documents": has_docs,
                "has_embeddings": has_embeddings,
                "has_search_engine": has_search_engine,
                "has_tfidf": has_tfidf,
                "document_count": len(self.documents),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def debug_search(self, query: str) -> Dict[str, Any]:
        """Debug version of search that shows detailed scoring"""
        print(f"\n=== DEBUGGING SEARCH FOR: '{query}' ===")
        
        debug_results = []
        for idx, doc in enumerate(self.documents):
            print(f"\n--- Document {idx}: {doc['name']} ---")
            is_relevant, scores = self._semantic_relevance_check(query, doc["content"][:1000], strict_threshold=0.15)
            print(f"Relevant: {is_relevant}")
            print(f"Scores: {scores}")
            
            debug_results.append({
                "document_name": doc["name"],
                "is_relevant": is_relevant,
                "scores": scores
            })
            
            if is_relevant:
                print("✓ PASSED relevance check")
            else:
                print("✗ FAILED relevance check")
        
        return {
            "query": query,
            "debug_results": debug_results
        }


# Global instance for reuse
_search_engine_instance = None

def get_search_engine() -> DocumentSearchEngine:
    """Get or create global search engine instance"""
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = DocumentSearchEngine()
    return _search_engine_instance

def reset_search_engine():
    """Reset global search engine instance"""
    global _search_engine_instance
    _search_engine_instance = None