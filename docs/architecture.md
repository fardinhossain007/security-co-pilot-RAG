# Architecture

## End-to-End Flow

```mermaid
flowchart TD
    A["PDF Documents (`data/raw`)"] --> B["Ingestion (`app/ingest.py`)"]
    B --> C["Text Cleaning + Chunking"]
    C --> D["Embeddings (`all-MiniLM-L6-v2`)"]
    D --> E["Chroma Vector Store (`data/chroma`)"]

    U["User Question (UI/API)"] --> R["Retriever (`max_marginal_relevance_search`)"]
    E --> R
    R --> P["Prompt Builder (`SYSTEM_PROMPT` + numbered chunks)"]
    P --> L["Local LLM (`ChatOllama`)"]
    L --> N["Post-process (`app/postprocess.py`)"]
    N --> O["Final Answer + Cited Sources"]

    O --> UI["Streamlit (`app/ui.py`)"]
    O --> API["FastAPI (`app/api.py`)"]
```

## Components

- Ingestion: extracts page text from PDFs, normalizes text, assigns metadata (`source_file`, `page_number`), then chunks for indexing.
- Retrieval: uses MMR to balance relevance and diversity across chunks.
- Generation: prompts local Ollama model to answer using only retrieved context.
- Post-processing: enforces answer shape and citation rules, removes low-value source-list outputs, and keeps only cited chunk mappings.
- Evaluation: local deterministic scripts measure citation/format compliance and lexical grounding proxies.

## Design Choices

- Local-first stack (Ollama + Chroma) keeps costs near zero.
- Explicit chunk numbering enables controllable citation references in output.
- Post-generation normalization improves robustness against prompt-format failures.
