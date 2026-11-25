"""
config.py 
---------
Every setting you might want to tweak lives here, in one place, so you never
have to hunt through multiple files to change a model name or chunk size.

If your senior asks "why did you pick these models?", the answer for all of
them is the same: they are the smallest widely-used option in their category
that still gives reasonable quality, which fits the "scale down" requirement
of the assignment. See docs/CONCEPTS.md for the reasoning behind each choice.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index_store")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_METADATA_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

# ---------------------------------------------------------------------------
# Stage 1: Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_WORDS = 200      # approx words per chunk (kept small = "scaled down")
CHUNK_OVERLAP_WORDS = 40    # words shared between consecutive chunks

# ---------------------------------------------------------------------------
# Stage 2: Embedding model (scaled down)
# ---------------------------------------------------------------------------
# 22M parameters, 384-dim vectors, runs fast even on CPU.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Stage 3: Retrieval
# ---------------------------------------------------------------------------
TOP_K_RETRIEVE = 5   # how many chunks to pull from the vector index initially

# ---------------------------------------------------------------------------
# Stage 4: Reranker (scaled down)
# ---------------------------------------------------------------------------
# Small cross-encoder, standard baseline for reranking experiments.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RERANK = 3     # how many chunks survive reranking and go to the LLM

# ---------------------------------------------------------------------------
# Stage 5: Generation (local LLM via Ollama)
# ---------------------------------------------------------------------------
OLLAMA_MODEL_NAME = "qwen2.5:3b"   # change to whatever you've pulled locally
OLLAMA_HOST = "http://localhost:11434"
