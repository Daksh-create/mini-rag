"""
reranker.py
-----------
STAGE 4: RERANKER (scaled down)

The initial retrieval step (retriever.py) is fast but approximate - it
compares a query vector to chunk vectors independently, which can miss
subtle relevance. A reranker looks at the query and each candidate chunk
TOGETHER (a "cross-encoder") and produces a much more accurate relevance
score, at the cost of being slower. Because it's slower, we only run it on
the small shortlist retrieval already narrowed down (e.g. top 5), not the
whole document set.

cross-encoder/ms-marco-MiniLM-L-6-v2 is a small (~22M param) cross-encoder -
the scaled-down standard baseline for this task.
""" 

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME, TOP_K_RERANK

_reranker = None  # lazy-loaded singleton


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"[reranker] Loading reranker model: {RERANKER_MODEL_NAME}")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(query: str, candidates, top_k: int = TOP_K_RERANK):
    """
    Takes the candidate chunks from retrieval and re-scores/re-orders them.
    Returns the top_k most relevant chunks according to the cross-encoder.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    # Quick manual test: run `python src/reranker.py`
    from embed_store import load_index
    from retriever import retrieve

    index, chunks = load_index()
    if index is None:
        print("No saved index found. Run `python src/embed_store.py` first.")
    else:
        test_query = "What is a small embedding model good for?"
        candidates = retrieve(test_query, index, chunks)
        reranked = rerank(test_query, candidates)
        print(f"Query: {test_query}\n")
        for r in reranked:
            print(f"[rerank={r['rerank_score']:.3f}] {r['source']} chunk {r['chunk_id']}")
            print(f"   {r['text'][:120]}...\n")
