# Setup Guide - Step by Step

Follow these in order. Every step includes what to run and how to tell it
worked.

## 1. Check Python

You need Python 3.9 or newer.

```bash
python3 --version
```

## 2. Create a virtual environment

Keeps this project's packages separate from everything else on your laptop.

```bash
cd mini-rag
python3 -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

Your terminal prompt should now show `(venv)` at the start of the line.

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs `sentence-transformers` (embedding + reranker models),
`faiss-cpu` (vector search), `ollama` (client for talking to your local
LLM), `numpy`, and `pypdf` (optional PDF support).

First time you run the pipeline, `sentence-transformers` will download the
embedding and reranker models automatically (a few hundred MB total) — this
needs an internet connection once, then everything runs fully offline.

## 4. Install Ollama (runs the local LLM)

Ollama is the easiest way to run a small LLM locally with GPU acceleration
already handled for you.

1. Download and install from **https://ollama.com/download** (has installers
   for Windows/Mac/Linux)
2. Verify it installed:
   ```bash
   ollama --version
   ```
3. Pull a small model (this downloads it once, ~2GB for a 3B model):
   ```bash
   ollama pull qwen2.5:3b
   ```
4. Ollama runs a background server automatically after install. You can
   confirm it's running with:
   ```bash
   ollama list
   ```
   which should show `qwen2.5:3b` in the list.

If you'd rather use an even smaller/faster model for a low-VRAM laptop, try:
```bash
ollama pull qwen2.5:1.5b
```
and change `OLLAMA_MODEL_NAME` in `src/config.py` to match.

## 5. Run the pipeline

```bash
python src/main.py
```

What happens:
- First run: loads documents from `data/`, chunks them, downloads/loads the
  embedding model, builds a FAISS index, and saves it to `index_store/`.
- Then it drops you into a chat loop — type a question, get an answer
  grounded in the sample documents.

Try asking:
- "What is RAG?"
- "Why use a small embedding model?"
- "What's the difference between fine-tuning and LoRA?"

Type `exit` to quit.

## 6. Rebuilding the index

If you edit/add files in `data/`, force a rebuild:

```bash
python src/main.py --rebuild
```

## 7. Testing each stage individually (useful for debugging)

Every file in `src/` can be run on its own to test just that stage:

```bash
python src/ingest.py       # confirms documents load correctly
python src/chunker.py      # shows how documents get split
python src/embed_store.py  # builds the index from scratch, prints vector count
python src/retriever.py    # runs one test retrieval query
python src/reranker.py     # runs retrieval + reranking on a test query
python src/generator.py    # runs the full pipeline once, prints one answer
```

Run these **in order** the first time — later ones assume the index already
exists (except `embed_store.py`, which builds it).

## Common issues

| Problem | Likely fix |
|---|---|
| `ModuleNotFoundError: No module named 'faiss'` | You forgot to activate the venv, or `pip install -r requirements.txt` failed — re-run it |
| Ollama connection errors | Ollama isn't running — reinstall or restart it; check `ollama list` works |
| Very slow first run | Normal — it's downloading the embedding/reranker models (one-time only) |
| Out of GPU memory | Switch to a smaller Ollama model (`qwen2.5:1.5b` or `qwen2.5:0.5b`) in `config.py` |
| `No saved index found` when testing retriever.py directly | Run `python src/embed_store.py` first to build the index |
