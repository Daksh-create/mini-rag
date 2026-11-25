"""
generator.py
------------
STAGE 5: GENERATION
 
Takes the final, reranked chunks and the user's question, builds a prompt
that tells the LLM to answer using ONLY that context, and sends it to a
small local model running via Ollama (e.g. qwen2.5:3b). Ollama must already
be running in the background (it starts a local server on port 11434 once
installed) - see docs/SETUP.md.

Note: this stage does NOT train or modify the model in any way. The model's
weights are frozen; we are only changing what we show it at request time.
That distinction is the whole point of RAG vs. fine-tuning - see
docs/CONCEPTS.md if that's still fuzzy.
"""

import ollama

from config import OLLAMA_MODEL_NAME, OLLAMA_HOST

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know - do not make things up.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, context_chunks) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, context_chunks, model: str = OLLAMA_MODEL_NAME) -> str:
    prompt = build_prompt(question, context_chunks)

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    # Quick manual test: run `python src/generator.py`
    # (requires Ollama running + the model already pulled)
    from embed_store import load_index
    from retriever import retrieve
    from reranker import rerank

    index, chunks = load_index()
    if index is None:
        print("No saved index found. Run `python src/embed_store.py` first.")
    else:
        test_query = "What is a small embedding model good for?"
        candidates = retrieve(test_query, index, chunks)
        top_chunks = rerank(test_query, candidates)
        answer = generate_answer(test_query, top_chunks)
        print(f"Q: {test_query}\n\nA: {answer}")
