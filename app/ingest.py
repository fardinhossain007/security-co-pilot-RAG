# app/ingest.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


RAW_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "security_corpus_v1"

# Good default embedding model (fast, solid quality, free)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def clean_text(t: str) -> str:
    """Light cleanup to reduce weird spacing and broken hyphenation."""
    t = t.replace("\x00", " ")
    # join hyphenated line breaks: "cyber-\nsecurity" -> "cybersecurity"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def load_pdfs(pdf_paths: List[Path]):
    docs = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(str(pdf))
        loaded = loader.load()  # one Document per page
        for d in loaded:
            d.page_content = clean_text(d.page_content)
            # Ensure consistent metadata keys
            d.metadata["source_file"] = pdf.name
            # PyPDFLoader uses "page" 0-indexed
            d.metadata["page_number"] = int(d.metadata.get("page", 0)) + 1
        docs.extend(loaded)
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,        # good starting point for policy docs
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Recreate fresh index each time to avoid duplicates (simple + safe for beginners)
    if CHROMA_DIR.exists():
        # delete old index
        for p in CHROMA_DIR.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(CHROMA_DIR.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    db.persist()
    return db


def main():
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found in data/raw. Put your PDFs there first.")

    print("Found PDFs:")
    for p in pdfs:
        print(" -", p.name)

    print("\nLoading PDFs (page-by-page)...")
    docs = load_pdfs(pdfs)
    print(f"Loaded {len(docs)} pages")

    print("\nSplitting into chunks...")
    chunks = split_docs(docs)
    print(f"Created {len(chunks)} chunks")

    # quick sanity stats
    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chars = total_chars / max(len(chunks), 1)
    print(f"Avg chunk chars: {avg_chars:.1f}")

    print("\nBuilding Chroma vectorstore (this may take a minute)...")
    _ = build_vectorstore(chunks)
    print(f"✅ Done. Vector DB saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
