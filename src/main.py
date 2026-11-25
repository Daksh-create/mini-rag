"""
main.py 
-------
THE PIPELINE. This is the file you actually run.

query -> retrieve -> rerank -> generate -> answer

It builds (or loads) the vector index once, then drops you into an
interactive loop where you can ask questions about whatever is in data/.
Each stage's latency is printed so you can see where time is being spent -
useful for the "scaled down vs full scale" comparison your senior mentioned.

Usage:
    python src/main.py              # build index if missing, then chat
    python src/main.py --rebuild    # force re-embedding, even if index exists 
"""

import sys
import time

from ingest import load_documents
from chunker import chunk_documents
from embed_store import build_index, load_index
from retriever import retrieve
from reranker import rerank
from generator import generate_answer


def get_or_build_index(force_rebuild: bool = False):
    if not force_rebuild:
        index, chunks = load_index()
        if index is not None:
            print(f"[main] Loaded existing index with {index.ntotal} chunks.")
            return index, chunks

    print("[main] Building index from documents in data/ ...")
    docs = load_documents()
    chunks = chunk_documents(docs)
    index, chunks = build_index(chunks)
    return index, chunks


def answer_question(question: str, index, chunks):
    timings = {}

    t0 = time.perf_counter()
    candidates = retrieve(question, index, chunks)
    timings["retrieve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    top_chunks = rerank(question, candidates)
    timings["rerank"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    answer = generate_answer(question, top_chunks)
    timings["generate"] = time.perf_counter() - t0

    return answer, top_chunks, timings


def print_timings(timings):
    parts = [f"{stage}={seconds:.2f}s" for stage, seconds in timings.items()]
    print(f"[timing] {' | '.join(parts)}")


def main():
    force_rebuild = "--rebuild" in sys.argv
    index, chunks = get_or_build_index(force_rebuild=force_rebuild)

    print("\nMini-RAG ready. Ask a question about the documents in data/.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        answer, sources, timings = answer_question(question, index, chunks)

        print(f"\nAssistant: {answer}\n")
        print("Sources used:")
        for s in sources:
            print(f"  - {s['source']} (chunk {s['chunk_id']}, "
                  f"rerank_score={s['rerank_score']:.3f})")
        print_timings(timings)
        print()


if __name__ == "__main__":
    main()
