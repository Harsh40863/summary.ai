# ---------------------------
# Must Be First Streamlit Command
# ---------------------------
import streamlit as st
st.set_page_config(page_title="SUMMARY.AI", layout="wide", page_icon="📚")

# ---------------------------
# Imports
# ---------------------------
import os
from rag.langchain_tools import GoogleSearchTool
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

# Custom modules
from text_preprocessing.cleaner import clean_text
from embedding.embedder import get_embeddings
from search.vector_search import VectorSearchEngine
from summarization.summarizer import flan_summarize_with_query as summarize_text_with_query
from rag.rag_engine import generate_answer_with_rag

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# ---------------------------
# Google Search Function
# ---------------------------
def google_search(query, api_key, cse_id, num=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": api_key, "cx": cse_id, "num": num}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# ---------------------------
# Title and Query UI
# ---------------------------
st.title("📚 Smart AI-Powered Document Search Plus")

# ---------------------------
# Download tokenizer
# ---------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
# ---------------------------
# Upload Interface
# ---------------------------
st.sidebar.header("📤 Upload Files")
from uploadmdb.db_ingest import upload_file_to_db

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/DOCX/PPTX/TXT/JPG/PNG",
    accept_multiple_files=True,
    type=["pdf","docx","pptx","txt","jpg","jpeg","png"]
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        result = upload_file_to_db(uploaded_file)
        st.sidebar.success(f"{uploaded_file.name}: {result}")
    

# ---------------------------
# Load and Preprocess
# ---------------------------
from uploadmdb.db_ingest import fetch_documents_from_db
documents = fetch_documents_from_db()

if not documents:
    st.error("No valid documents found in the folder. Please upload or check content.")
    st.stop()

doc_texts = [clean_text(doc["content"]) for doc in documents]
doc_embeddings = get_embeddings(doc_texts)
index_to_docinfo = {i: doc for i, doc in enumerate(documents)}

n_clusters = min(len(documents), 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(doc_embeddings)
topic_labels = kmeans.labels_

search_engine = VectorSearchEngine(dimension=doc_embeddings.shape[1])
search_engine.build_index(doc_embeddings, doc_texts)

# ---------------------------
# UI + Query Handling
# ---------------------------
if "prev_queries" not in st.session_state:
    st.session_state["prev_queries"] = []

if "last_summary" not in st.session_state:
    st.session_state["last_summary"] = ""
if "last_docname" not in st.session_state:
    st.session_state["last_docname"] = ""

lang_option = st.selectbox("Select Query Language (Coming Soon)", ["English", "Spanish", "French"])
voice_query = st.button("🎙 Use Voice Query (Coming Soon)")
if voice_query:
    st.info("Voice input is coming soon!")

if st.checkbox("Secure Search Mode (Demo)"):
    st.info("Secure search zones will be available in a future update.")

query = st.text_input("🔍 Enter your query:")
col1, col2, col3 = st.columns(3)
with col1:
    search_button = st.button("🔎 Search")
with col2:
    think_button = st.button("🧠 Think")
with col3:
    explore_button = st.button("🌐 Explore")

# Determine which action was taken
action = None
if search_button:
    action = "🔎 Search"
elif think_button:
    action = "🧠 Think"
elif explore_button:
    action = "🌐 Explore"

if query and action:
    st.session_state.prev_queries.append(query)
    clean_query = clean_text(query)
    query_embedding = get_embeddings([clean_query])

    # Step 1: Document-Aware Summary Creation (Shared for all 3 actions)
    relevant_results = []
    found_relevant = False

    for idx, doc in enumerate(documents):
        sentences = sent_tokenize(doc["content"])
        if not sentences:
            continue

        sentence_embeddings = get_embeddings(sentences)
        sim_scores = np.dot(sentence_embeddings, query_embedding.T).flatten()
        best_score = np.max(sim_scores)
        best_idx = np.argmax(sim_scores)

        threshold = 0.35
        if best_score < threshold:
            continue

        found_relevant = True
        start = max(0, best_idx - 1)
        end = min(len(sentences), best_idx + 2)
        relevant_chunk = " ".join(sentences[start:end])
        query_summary = generate_answer_with_rag(query, relevant_chunk, doc["name"])

        relevant_results.append({
            "doc": doc,
            "best_sentence": sentences[best_idx],
            "score": best_score,
            "relevant_chunk": relevant_chunk,
            "summary": query_summary
        })

    if not found_relevant:
        st.warning("🤷 No relevant documents found for your query.")
        st.stop()

    # Sort by score
    relevant_results = sorted(relevant_results, key=lambda x: x["score"], reverse=True)

    # Step 2: Handle Action Based on Selection
    for result in relevant_results:
        doc = result["doc"]
        summary = result["summary"]

        st.markdown(f"### 📄 {doc['name']}")
        st.markdown(f"**💡 Best Matching Sentence:** {result['best_sentence']} (Score: {result['score']:.2f})")

        if action == "🔎 Search":
            st.markdown(f"**📝 Summary:** {summary}")

        elif action == "🌐 Explore":
            st.markdown("**🌐 Relevant Web Content Based on Document Insight:**")
            try:
                enhanced_query = f"{query} - Context: {summary[:250]}"
                search_tool = GoogleSearchTool(api_key=GOOGLE_API_KEY2, cse_id=GOOGLE_CSE_ID)
                explore_results = search_tool.run(enhanced_query)
                st.markdown(explore_results)
            except Exception as e:
                st.error(f"Error fetching explore content: {e}")

        elif action == "🧠 Think":
            st.markdown("**🧠 Refined Insight Using Document + Query**")
            try:
                from rag.Google_Gen_AI import generate_answer_with_google
                refined_answer = generate_answer_with_google(query, summary, doc["name"])
                st.markdown(refined_answer)
            except Exception as e:
                st.error(f"Error during insight generation: {e}")
