# app/rag.py
from __future__ import annotations

from typing import List, Dict, Any

from langchain_ollama import ChatOllama

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.postprocess import CitationChunk, normalize_answer


CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "security_corpus_v1"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_K = 8  # retrieved chunks; balanced quality vs coverage


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return db


def retrieve(db: Chroma, question: str, k: int = DEFAULT_K) -> List[CitationChunk]:
    docs = db.max_marginal_relevance_search(question, k=k, fetch_k=max(20, k * 4))
    chunks: List[CitationChunk] = []
    for d in docs:
        meta = d.metadata or {}
        chunks.append(
            CitationChunk(
                text=d.page_content,
                source_file=str(meta.get("source_file", "unknown")),
                page_number=int(meta.get("page_number", meta.get("page", 0))) if meta else 0,
            )
        )
    return chunks

def format_context(chunks: List[CitationChunk]) -> str:
    """
    Provide the model with numbered chunks so it can cite them.
    We repeat the chunk id at the end to improve citation compliance.
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] SOURCE: {c.source_file} | PAGE: {c.page_number}\n"
            f"{c.text}\n"
            f"(chunk_id: {i})\n"
        )
    return "\n".join(parts)



SYSTEM_PROMPT = """You are a security policy assistant.

You MUST follow the format rules exactly.

GROUNDING RULE:
- Use ONLY the provided context. If the answer is not in the context, say:
  "I don't know based on the provided documents."

CITATION RULE (STRICT):
- Every bullet MUST end with citations in square brackets referencing context chunk numbers, e.g. [1] or [1][4].
- Do not write any factual statement without citations.
- If you cannot cite a claim, do not include it.

STYLE:
- Be concise.
- Avoid duplicates; merge overlapping points.
- Maximum 5 bullets.
"""

USER_PROMPT_TEMPLATE = """Question: {question}

Context (numbered chunks you MUST cite):
{context}

Return ONLY in this exact structure:

Answer:
- <bullet 1 ending with citations like [1] or [1][2]>
- <bullet 2 ending with citations like [3]>
- <... up to 5 bullets total>

Cited sources:
- [1] <SOURCE> p.<PAGE>
- [2] <SOURCE> p.<PAGE>
- ...

Important:
- Every bullet MUST end with citations.
- Only include citations that you actually used.
"""


def ask(
    question: str,
    k: int = DEFAULT_K,
    model: str = "llama3.1:8b",
    audit_mode: bool = False,
) -> Dict[str, Any]:
    """
    Local LLM (Ollama) RAG.

    audit_mode=True forces k=10 (more coverage, may be more redundant).
    """
    db = load_vectorstore()

    # Audit mode override (ignore slider value)
    if audit_mode:
        k = 10

    chunks = retrieve(db, question, k=k)
    context = format_context(chunks)

    llm = ChatOllama(model=model, temperature=0.1)

    msg = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE.format(question=question, context=context)),
        ]
    )

    raw_answer = msg.content if hasattr(msg, "content") else str(msg)
    answer, used_ids = normalize_answer(raw_answer, chunks)

    citation_map = [
        {"chunk": i, "file": c.source_file, "page": c.page_number}
        for i, c in enumerate(chunks, 1)
        if i in used_ids
    ]

    retrieved = [
        {
            "chunk": i,
            "file": c.source_file,
            "page": c.page_number,
            "preview": (c.text[:250] + "...") if len(c.text) > 250 else c.text,
        }
        for i, c in enumerate(chunks, 1)
    ]

    return {
        "answer": answer,
        "citations": citation_map,
        "retrieved": retrieved,
        "k": k,
        "audit_mode": audit_mode,
    }
