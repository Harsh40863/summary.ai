# summary.ai — RAG Evaluation Suite

A comprehensive, modular evaluation suite for the **summary.ai** document search & RAG platform.

---

## Folder Structure

```text
eval/
├── datasets/                 # Evaluation test sets
│   ├── retriever_golden_dataset.json   # Golden test cases for retriever & generator
│   └── guardrails_dataset.json         # Guardrails test cases (PII, toxicity, hallucination)
├── retriever/                # 1. Retriever Evaluation (Precision & Recall)
│   ├── eval_retriever.py     # DeepEval contextual precision/recall runner
│   ├── eval_grok.py          # Grok LLM Judge benchmark
│   ├── retriever_wrapper.py  # MMR + FAISS retrieval wrapper
│   └── run_eval.py           # Retriever CLI runner
├── generator/                # 2. Generator Evaluation (Faithfulness & Coverage)
│   ├── eval_generator.py     # Isolated generator benchmark (Groq / Grok judge)
│   └── generator_wrapper.py  # Prompt & generation wrapper
├── pipeline/                 # 3. End-to-End RAG Triad Evaluation
│   ├── eval_rag_pipeline.py  # Contextual Relevancy, Faithfulness, Answer Relevancy
│   └── __init__.py
├── safety/                   # 4. Guardrails & Observability Evaluation
│   ├── eval_langsmith.py     # LangSmith tracing, latency & safety runner
│   ├── langsmith_evaluators.py # PII, Toxicity, Hallucination evaluators
│   └── __init__.py
├── results/                  # Timestamped JSON results & scorecards
│   └── .gitkeep
├── __init__.py
├── __main__.py               # Main module runner (python -m eval)
├── harness.py                # Common evaluation harness & metrics helpers
├── run_evals.py              # Unified CLI runner for all evaluation suites
└── README.md                 # This file
```

---

## What is Being Evaluated

1. **Retriever ([`eval/retriever/`](retriever)):**
   - **Contextual Precision:** Signal-to-noise ratio in retrieved context.
   - **Contextual Recall:** Ground-truth coverage across retrieved chunks.
   - **F1 Score & Hit Rate:** Multi-metric retrieval accuracy.
2. **Generator ([`eval/generator/`](generator)):**
   - **Faithfulness:** Claims grounded strictly in reference context without hallucination.
   - **Answer Recall:** Coverage of critical ground-truth facts.
3. **End-to-End RAG Triad ([`eval/pipeline/`](pipeline)):**
   - **Contextual Relevancy:** Relevance of retrieved context to query.
   - **Faithfulness:** Zero hallucination in final response.
   - **Answer Relevancy:** Conciseness and directness of answer to query.
4. **Safety & Guardrails ([`eval/safety/`](safety)):**
   - **Trace Linkage:** Span-level binding from query to retrieval to generation.
   - **Anti-Hallucination & Fabrication:** Verification of claims against source context.
   - **Toxicity Safety:** Zero toxic or harmful content.
   - **PII Protection:** Zero leakage of sensitive secrets, emails, keys, and credentials.

---

## Quick Start & Usage

Make sure your virtual environment is active and dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Run the Unified Suite (All Benchmarks)

```bash
# Run all suites (Retriever + RAG Triad + LangSmith Safety):
python -m eval

# Fast mode (~45 seconds on curated representative samples):
python -m eval --mode quick

# Filter by document category:
python -m eval --category financial_report
```

### 2. Run Component-Specific Evaluations

```bash
# RAG Triad Pipeline Evaluation:
python -m eval.pipeline.eval_rag_pipeline
python -m eval.pipeline.eval_rag_pipeline --category financial_report --threshold 0.75

# Retriever Evaluation (Grok LLM Judge):
python -m eval.retriever.eval_grok
python -m eval.retriever.eval_grok --category legal_contract

# Retriever Evaluation (DeepEval):
python -m eval.retriever.run_eval --category policy_document

# Generator Evaluation (Groq / Grok LLM Judge):
python -m eval.generator.eval_generator --provider groq --model qwen/qwen3.8-27b
python -m eval.generator.eval_generator --category technical_spec

# LangSmith Tracing & Safety Guardrails:
python -m eval.safety.eval_langsmith --dataset guardrails --upload
```

---

## Environment Variables

Configure required API keys in `.env` or export in your shell:

```bash
# LLM Judges & Generation
export GROQ_API_KEY="gsk_..."
export XAI_API_KEY="xai-..."
export GEMINI_API_KEY="..."

# LangSmith Tracing (Optional)
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_PROJECT="summary.ai"
export LANGSMITH_TRACING="true"
```

All evaluation scorecards and JSON reports are automatically saved to [`eval/results/`](results/).
