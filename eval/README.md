# summary.ai — RAG Evaluation Suite

A standalone evaluation suite for the **summary.ai** document search & RAG platform.  
Component-level evaluation is built first; the retriever is evaluated for **precision** and **recall** using the [deepeval](https://docs.deepeval.com) library.

---

## Folder Structure

```
eval/
├── datasets/
│   └── retriever_golden_dataset.json   # Hand-crafted golden test cases for the retriever
├── retriever/
│   ├── retriever_wrapper.py            # Wraps the app's MMR + FAISS retrieval pipeline
│   ├── eval_retriever.py               # deepeval precision / recall runner
│   └── run_eval.py                     # CLI wrapper for eval_retriever
├── harness.py                          # load_goldens, summarize_by_metric, print_summary
├── results/                            # Auto-created; timestamped JSON results saved here
├── conftest.py                         # Shared pytest fixtures (golden dataset, retriever factory)
└── README.md                           # This file
```

---

## What is Being Evaluated

The retriever in `backend/logic.py` uses a **2-stage hierarchical approach**:

1. **Stage 1 (doc-level):** FAISS `IndexFlatIP` compares query embedding vs whole-document embeddings, selects top-2 docs.
2. **Stage 2 (chunk-level):** Each selected doc is split into 10-sentence chunks; a temporary FAISS index is built and searched for the best chunk.

**Metrics:**
| Metric | What it measures |
|---|---|
| **Contextual Precision** | Of the retrieved chunks, how many are actually relevant to the query? |
| **Contextual Recall** | Of all truly relevant chunks, how many did the retriever return? |

Both metrics are computed by deepeval using an LLM judge (OpenAI GPT-4 by default).

---

## Setup

Make sure the venv is active and deepeval is installed:

```bash
cd /path/to/summary.ai
source venv/bin/activate
pip install deepeval          # already done
```

Set the LLM judge API key (deepeval uses OpenAI by default):

```bash
export OPENAI_API_KEY=sk-...
```

Or evaluate using **Grok (xAI)**:

```bash
export XAI_API_KEY=xai-...
export GROK_MODEL=grok-2-latest    # optional, defaults to grok-2-latest
```

Or Gemini:

```bash
export DEEPEVAL_LLM_MODEL=gemini/gemini-1.5-flash
export GOOGLE_API_KEY=...
```

---

## Running the Evaluation

### Option 1: Fast Standalone Retrieval Eval with Grok Judge (Precision & Recall)

```bash
# Run all golden cases with Grok judge and retry wrapper
python eval_retrieval_grok.py

# Or via submodule path
python eval/retriever/eval_grok.py

# Filter by category or custom threshold
python eval_retrieval_grok.py --category financial_report
python eval_retrieval_grok.py --threshold 0.75
```

### Option 2: Generator Evaluation with Groq / Grok Judge (Precision & Recall)

```bash
# Run all golden cases for Generator with Groq judge
python eval_generator_groq.py

# Or via submodule path
python eval/generator/eval_generator.py

# Use specific provider, category, or threshold
python eval_generator_groq.py --provider groq --model qwen/qwen3.8-27b
python eval_generator_groq.py --provider grok --model grok-2-latest
python eval_generator_groq.py --category financial_report
python eval_generator_groq.py --threshold 0.75
```

### Option 3: DeepEval Retriever Runner

```bash
# Run all 12 test cases
python eval/retriever/run_eval.py

# Run with Grok judge via DeepEval
python eval/retriever/run_eval.py --judge-model grok-2-latest

# Run only financial_report category
python eval/retriever/run_eval.py --category financial_report

# Skip negative tests (out-of-domain queries)
python eval/retriever/run_eval.py --skip-negative

# Set custom pass/fail thresholds
python eval/retriever/run_eval.py --precision-threshold 0.6 --recall-threshold 0.6
```

### Option 4: End-to-End Pipeline Evaluation (RAG Triad & Token-Based Chunking)

Evaluates the live system across the **RAG Triad** (`Contextual Relevancy`, `Faithfulness`, `Answer Relevancy`) using production-grade **Token-Based Chunking** (`CHUNK_SIZE_TOKENS=60`, `CHUNK_OVERLAP_TOKENS=20` via `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with `cl100k_base`):

```bash
# Run end-to-end pipeline evaluation on financial_report
python eval_rag_pipeline.py --category financial_report

# Run across all categories with Groq judge
python eval_rag_pipeline.py --threshold 0.70

# Filter by custom category
python eval_rag_pipeline.py --category legal_contract
```

#### Token-Based Chunking Benchmark Results:
| Metric | Average Score (%) | Pass Rate (%) | Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Faithfulness** | **100.0%** | **100.0%** | $\ge 70\%$ | **PASS** |
| **Answer Relevancy** | **88.33%** | **83.33%** | $\ge 70\%$ | **PASS** |
| **Contextual Relevancy** | **63.81%** | **50.00%** | $\ge 70\%$ | **IMPROVED** (from $\sim 35\text{--}40\%$) |

Results are printed to the console and saved as JSON in `eval/results/`.

---

## Golden Dataset

Located at [`datasets/retriever_golden_dataset.json`](datasets/retriever_golden_dataset.json).

**12 test cases** covering realistic document types the app handles:

| Category | # Cases | What it tests |
|---|---|---|
| `factual_lookup` | 2 | Simple direct-answer queries from research papers |
| `technical_detail` | 1 | Specific numerical / technical facts |
| `financial_report` | 2 | Revenue, profit, division queries |
| `policy_document` | 2 | HR policy lookup (sick leave, maternity) |
| `legal_contract` | 2 | Contract termination clauses, confidentiality |
| `technical_spec` | 2 | System requirements, GPU/CUDA queries |
| `multi_doc_retrieval` | 1 | Query that spans two documents |
| `negative_test` | 1 | Out-of-domain query that should return no relevant chunks |

Each test case contains:
- `query` – the user's search query  
- `input_documents` – inline document content (no MongoDB required for eval)  
- `expected_chunks` – ground-truth chunks with relevance labels (`high` / `medium`)  
- `top_k`, `expected_threshold`, `min_relevant_in_top_k` – retrieval parameters  

---

## Extending the Suite

**Add more retriever test cases:**  
Append entries to `datasets/retriever_golden_dataset.json` following the existing schema.

**Add a new component (e.g. generator, end-to-end):**  
Create a new folder under `eval/` (e.g. `eval/generator/`) with its own `run_eval.py` and dataset.

---

## Output Example

```
============================================================
  summary.ai — Retriever Evaluation Suite
============================================================

📂 Loaded 12 test cases from dataset

🔍 Running retrieval on test cases...
  ✓ Built test case TC001 – factual_lookup | retrieved 2 chunks
  ...

📊 Evaluating 12 test cases with deepeval...

Evaluating test cases: 100%|████████| 12/12

======================================================================
 Metric              Score    Threshold    Pass
----------------------------------------------------------------------
 Contextual Precision  0.82     0.50        ✓
 Contextual Recall     0.75     0.50        ✓
======================================================================

📄 Results saved → eval/results/retriever_results_20241008_152300.json
```
