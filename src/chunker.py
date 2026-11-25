"""
chunker.py
----------
STAGE 1: CHUNKING (scaled down) 

Splits long documents into small overlapping chunks. We use simple
word-count-based splitting instead of expensive semantic or LLM-based
chunking - this is the "scaled down" version of chunking your senior asked
for. It's fast, has zero dependencies beyond the standard library, and is
good enough for a learning-scale project.

Why overlap? If a sentence gets cut in half at a chunk boundary, overlap
means the full sentence still appears intact in at least one chunk, so we
don't lose meaning at the edges.
"""

from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS):
    """
    Splits `text` into a list of overlapping word-based chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)  # avoid infinite loop if overlap >= chunk_size

    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def chunk_documents(documents):
    """
    Takes the list of documents from ingest.load_documents() and returns a
    flat list of chunk dicts:
        [{"source": "sample1.txt", "chunk_id": 0, "text": "..."}, ...]
    """
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": piece,
            })
    return all_chunks


if __name__ == "__main__":
    # Quick manual test: run `python src/chunker.py`
    from ingest import load_documents
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} document(s).")
    print("\nExample chunk:")
    print(chunks[0])
