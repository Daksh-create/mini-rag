"""
retriever.py
------------
STAGE 3: RETRIEVAL (scaled down)

Given a user query, embeds it with the SAME embedding model used for the
chunks, then searches the FAISS index for the most similar chunks. This is
"scaled down" in the sense that it's a single flat in-memory index doing an
exact search - fine for thousands of chunks on a laptop, though it wouldn't
scale to a production system with millions of documents (which would need an
approximate-search index or a real vector DB).
"""

from config import TOP_K_RETRIEVE 
from embed_store import embed_texts


def retrieve(query: str, index, chunks, top_k: int = TOP_K_RETRIEVE):
    """
    Returns the top_k most similar chunks to `query`, each with a similarity
    score attached.
    """
    query_vector = embed_texts([query])
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS returns -1 if there aren't enough vectors
        chunk = dict(chunks[idx])  # copy so we don't mutate the original
        chunk["retrieval_score"] = float(score)
        results.append(chunk)

    return results


if __name__ == "__main__":
    # Quick manual test: run `python src/retriever.py`
    from embed_store import load_index

    index, chunks = load_index()
    if index is None:
        print("No saved index found. Run `python src/embed_store.py` first.")
    else:
        test_query = "What is a small embedding model good for?"
        results = retrieve(test_query, index, chunks)
        print(f"Query: {test_query}\n")
        for r in results:
            print(f"[{r['retrieval_score']:.3f}] {r['source']} chunk {r['chunk_id']}")
            print(f"   {r['text'][:120]}...\n")
