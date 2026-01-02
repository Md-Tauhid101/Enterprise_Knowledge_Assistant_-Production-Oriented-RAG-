
# Enterprise Knowledge Assistant (Production-Oriented RAG)

A **failure-aware, evaluation-driven Enterprise Knowledge Assistant** built to answer questions from internal documents **only when evidence supports the answer** — otherwise it explicitly refuses.

This is **not** a “Chat with PDF” demo.  
It models real enterprise constraints: messy data, retrieval failure, validation, and auditability.

---

## What Problem This Solves

Enterprises don’t lack LLMs.  
They lack **trustworthy information systems**.

Common failures in naive RAG systems:
- Confident hallucinations
- Silent retrieval failure
- No provenance or traceability
- No evaluation of correctness

This project solves that by:
- Separating **search** from **truth**
- Enforcing **evidence validation**
- Supporting **explicit refusal**
- Measuring retrieval and answer quality

---

## Core Capabilities

- Ingests messy PDFs, DOCs, CSVs
- Cleans and chunks documents deterministically
- Stores ground truth in Postgres
- Uses hybrid retrieval (Dense + BM25)
- LangGraph-based orchestration (stateful, guarded)
- Evidence-bound answer generation
- Explicit refusal when unsupported
- Offline evaluation (Recall@K, Faithfulness, Refusal correctness)

---

## System Architecture (High-Level)

Ingestion (Offline)
→ Clean & Chunk  
→ Store in Postgres  
→ Build Vector + BM25 Index  

Query-Time (LangGraph)
→ Intent Check  
→ Guarded Query Rewrite  
→ Hybrid Retrieval  
→ Evidence Validation  
→ Answer OR Refuse  

Evaluation (Offline)
→ Recall@K  
→ Faithfulness  
→ Refusal Accuracy  

---

## Tech Stack

- Python 3.10
- LangGraph
- PostgreSQL
- FAISS
- BM25
- uv (package manager)

---

## Installation (Using `uv`)

### Prerequisites
- Python 3.10
- PostgreSQL
- git

### Clone Repository
```bash
git clone https://github.com/Md-Tauhid101/Enterprise_Knowledge_Assistant_-Production-Oriented-RAG-.git
```

### Install uv
```bash
pip install uv
```

### Create Virtual Environment
```bash
uv venv
```

Activate:
- Windows: `.venv\Scripts\activate`
- Linux/macOS: `source .venv/bin/activate`

### Install Dependencies
```bash
uv sync
```

---

## Running the Project

### Ingest Documents
```bash
uv run -m offline_pipeline
```

### Build Sparse Indexes
```bash
uv run -m indexes.sparse_index
```

### Run Query Flow
```bash
uv run -m online_pipeline
```

---

## Refusal Behavior

The system refuses when:
- Evidence is missing
- Sources conflict
- Retrieval is insufficient

This is **intentional**.

---