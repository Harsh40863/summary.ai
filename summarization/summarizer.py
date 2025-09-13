from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the Flan-T5 model
MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def flan_summarize_with_query(context, query):
    """
    Uses the Flan-T5 model to generate a summary that is focused on answering the query
    using the provided context.
    """
    prompt = f"Summarize the following text in a way that directly answers the question: '{query}'\n\nContext:\n{context}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary