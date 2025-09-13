# rag/rag_engine.py
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_answer_with_rag(query, summary, doc_name):
    """
    Generate a deeper, more refined answer using the provided query, summary,
    and document name, acting as a mini RAG system. The prompt instructs the model
    to generate a detailed answer using only the supplied context.
    """
    prompt = (
        f"You are a highly knowledgeable assistant tasked with answering questions using only the information provided below.\n\n"
        f"The user has asked the following question:\n\"{query}\"\n\n"
        f"Here is a context summary extracted from the document titled \"{doc_name}\":\n{summary}\n\n"
        "Using this context, provide a clear, accurate, and thoughtful answer. "
        "If the answer is not directly stated, use reasoning and inference based on the context to give the best possible explanation. "
        "Do not include any external information. Keep your answer grounded strictly in the provided summary."
        )


    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=300,
        min_length=80,
        num_beams=5,
        temperature=0.7,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer