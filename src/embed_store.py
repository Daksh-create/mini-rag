"""
embed_store.py 
--------------
STAGE 2: EMBEDDING MODEL (scaled down) + vector storage

Uses all-MiniLM-L6-v2 (22M params) via sentence-transformers to convert each
chunk into a 384-dimensional vector, then stores those vectors in a FAISS
index for fast similarity search. FAISS runs fully in-process on your
laptop - no external server needed, which is what makes this "scaled down"
compared to a hosted vector database.

The index and chunk metadata are saved to disk (index_store/) so you only
need to re-embed when your documents actually change.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
)

_embedder = None  # lazy-loaded singleton so we only load the model once


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[embed_store] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed_texts(texts):
    """Converts a list of strings into a numpy array of embedding vectors."""
    embedder = get_embedder()
    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 20,
        normalize_embeddings=True,  # so we can use inner product == cosine similarity
    )
    return vectors.astype("float32")


def build_index(chunks):
    """
    Builds a FAISS index from a list of chunk dicts (see chunker.py output)
    and saves both the index and the chunk metadata to disk.
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product == cosine sim (vectors are normalized)
    index.add(vectors)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[embed_store] Indexed {len(chunks)} chunks -> {FAISS_INDEX_PATH}")
    return index, chunks


def load_index():
    """Loads a previously saved FAISS index + chunk metadata from disk."""
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_METADATA_PATH)):
        return None, None
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_METADATA_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


if __name__ == "__main__":
    # Quick manual test: run `python src/embed_store.py`
    from ingest import load_documents
    from chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    index, chunks = build_index(chunks)
    print(f"Index contains {index.ntotal} vectors of dimension {index.d}")
