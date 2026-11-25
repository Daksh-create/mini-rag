"""
ingest.py 
---------
Loads raw source documents from the data/ folder.

This is deliberately simple: it reads every .txt file (and .pdf file, if
pypdf is installed) in DATA_DIR and returns a list of documents, each with
its source filename and full text. Chunking happens separately in chunker.py
so each file has one responsibility.
"""

import os
from config import DATA_DIR


def _read_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf(filepath: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is not installed. Run: pip install pypdf   "
            "(or just use .txt files in data/ instead)"
        )
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(data_dir: str = DATA_DIR):
    """
    Returns a list of dicts: [{"source": "sample1.txt", "text": "..."}, ...]
    """
    documents = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(".txt"):
            text = _read_txt(filepath)
        elif filename.lower().endswith(".pdf"):
            text = _read_pdf(filepath)
        else:
            continue  # skip unsupported file types

        if text.strip():
            documents.append({"source": filename, "text": text})

    if not documents:
        raise ValueError(
            f"No .txt or .pdf documents found in {data_dir}. "
            "Add some files before running the pipeline."
        )

    return documents


if __name__ == "__main__":
    # Quick manual test: run `python src/ingest.py` to sanity-check loading.
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s):")
    for d in docs:
        print(f"  - {d['source']}  ({len(d['text'])} characters)")
