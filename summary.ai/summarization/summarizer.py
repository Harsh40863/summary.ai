from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
_tokenizer = None
_model = None


def _get_flan_t5():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _tokenizer, _model

def flan_summarize_with_query(context, query):
    """
    Uses the Flan-T5 model to generate a summary that is focused on answering the query
    using the provided context.
    """
    tokenizer, model = _get_flan_t5()
    prompt = f"Summarize the following text in a way that directly answers the question: '{query}'\n\nContext:\n{context}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary
