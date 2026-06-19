# rag/rag_engine.py
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dotenv import load_dotenv

load_dotenv()

model_name = "google/flan-t5-base"
_tokenizer = None
_model = None


def _get_rag_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = T5Tokenizer.from_pretrained(model_name)
    if _model is None:
        _model = T5ForConditionalGeneration.from_pretrained(model_name)
    return _tokenizer, _model


def _fallback_answer(query, summary, doc_name):
    try:
        from rag.Google_Gen_AI import generate_answer_with_google
        return generate_answer_with_google(query, summary, doc_name, action_type="search")
    except Exception:
        return (
            f"**Context from '{doc_name}':**\n\n"
            f"{summary}"
        )

def generate_answer_with_rag(query, summary, doc_name):
    """
    Generate a deeper, more refined answer using the provided query, summary,
    and document name, acting as a mini RAG system. The prompt instructs the model
    to generate a detailed answer using only the supplied context.
    """
    if os.getenv("ENABLE_FLAN_T5", "false").lower() != "true":
        return _fallback_answer(query, summary, doc_name)

    prompt = (
        f"You are a highly knowledgeable assistant tasked with answering questions using only the information provided below.\n\n"
        f"The user has asked the following question:\n\"{query}\"\n\n"
        f"Here is a context summary extracted from the document titled \"{doc_name}\":\n{summary}\n\n"
        "Using this context, provide a clear, accurate, and thoughtful answer. "
        "If the answer is not directly stated, use reasoning and inference based on the context to give the best possible explanation. "
        "Do not include any external information. Keep your answer grounded strictly in the provided summary."
        )


    try:
        tokenizer, model = _get_rag_model()
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        outputs = model.generate(
            **inputs,
            max_length=1000,
            min_length=80,
            num_beams=5,
            temperature=0.7,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"{_fallback_answer(query, summary, doc_name)}\n\nModel generation skipped: {e}"
