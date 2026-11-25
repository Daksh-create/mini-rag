# Mini RAG Pipeline

A lightweight, modular Retrieval-Augmented Generation (RAG) pipeline designed to run locally on consumer hardware using **FAISS**, **Sentence-Transformers**, and **Ollama**.

---

## Architecture Overview

The system implements a five-stage **retrieve → rerank → generate** workflow:

1. **Ingestion (`src/ingest.py`)** — Loads `.txt` files (and `.pdf` files via `pypdf`) from the `data/` directory.
2. **Chunking (`src/chunker.py`)** — Splits documents into overlapping word-count chunks (`200` words per chunk with a `40`-word overlap by default) to preserve boundary context.
3. **Embedding & Vector Store (`src/embed_store.py`)** — Encodes chunks into `384`-dimensional normalized vectors using `sentence-transformers/all-MiniLM-L6-v2` and persists an exact inner-product FAISS index (`IndexFlatIP`) alongside chunk metadata.
4. **Retrieval & Reranking (`src/retriever.py`, `src/reranker.py`)** — Performs fast vector similarity search to fetch the top `5` candidates, then re-scores `(query, chunk)` pairs jointly using `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top `3` passages.
5. **Generation (`src/generator.py`)** — Injects the top reranked chunks with source attributions into a grounded prompt and generates answers locally via Ollama (`qwen2.5:3b`).

---

## Project Structure

```text
mini-rag/
├── data/                     # Source .txt and .pdf documents
├── src/
│   ├── config.py             # Centralized paths, model names, and hyperparameters
│   ├── ingest.py             # Document loader (.txt and .pdf)
│   ├── chunker.py            # Overlapping word-based text chunker
│   ├── embed_store.py        # Embedding generator and FAISS index builder/loader
│   ├── retriever.py          # Top-k dense vector similarity search
│   ├── reranker.py           # Cross-encoder candidate reranker
│   ├── generator.py          # Context-grounded prompt formatter and Ollama client
│   └── main.py               # Interactive CLI pipeline with per-stage latency tracking
├── docs/
│   └── SETUP.md              # Step-by-step installation and troubleshooting guide
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama and pull the LLM

Download and install [Ollama](https://ollama.com/download), ensure the local server is running, and pull the default model:

```bash
ollama pull qwen2.5:3b
```

*(For lower VRAM setups, you can pull a smaller model such as `qwen2.5:1.5b` and update `OLLAMA_MODEL_NAME` in `src/config.py`.)*

---

## Usage

### Run the interactive pipeline

```bash
python src/main.py
```

- **First run:** Loads documents from `data/`, builds the FAISS index, saves it to `index_store/`, and launches the interactive Q&A prompt.
- **Subsequent runs:** Reuses the saved index automatically. To force re-indexing after updating `data/`, run:

```bash
python src/main.py --rebuild
```

### Run individual pipeline stages

Each module in `src/` includes a standalone test entrypoint:

```bash
python src/ingest.py        # Verify document loading
python src/chunker.py       # Inspect document chunking
python src/embed_store.py   # Build the FAISS index and verify vector count
python src/retriever.py     # Test top-k vector retrieval
python src/reranker.py      # Test cross-encoder reranking
python src/generator.py     # Run an end-to-end single-query test
```

---

## Configuration

All pipeline settings are managed in `src/config.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `CHUNK_SIZE_WORDS` | `200` | Target number of words per chunk |
| `CHUNK_OVERLAP_WORDS` | `40` | Overlapping words between consecutive chunks |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Bi-encoder model for document and query embeddings |
| `TOP_K_RETRIEVE` | `5` | Number of candidate chunks retrieved from FAISS |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model used to rerank candidates |
| `TOP_K_RERANK` | `3` | Number of reranked chunks passed to the LLM |
| `OLLAMA_MODEL_NAME` | `qwen2.5:3b` | Local Ollama model used for answer generation |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama server URL |
````[cite: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]