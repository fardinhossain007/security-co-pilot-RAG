# Security Copilot RAG

[![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](./README.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](./requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)](./README.md)

Local-first Retrieval-Augmented Generation (RAG) assistant for cybersecurity policy documents with citation-aware answers.

Supported corpus:
- NIST CSF 2.0 (`NIST.CSWP.29`)
- NIST SP 800-61r3
- OWASP Top 10 for LLM Applications 2025

## TL;DR

- This is **v1** of a local-first security policy RAG copilot.
- It ingests trusted cybersecurity PDFs, retrieves relevant chunks with Chroma/MMR, and generates citation-aware answers with Ollama.
- It includes a deterministic local evaluation pipeline and tests for answer/citation post-processing behavior.
- Current benchmark indicates `k=10` performs better than `k=8` on citation and grounding proxy metrics.
- Next versions will add hosted model support (including OpenAI), larger document coverage, and stronger evaluation rigor.

## Overview

This project is designed as a production-style portfolio project for a first RAG system:
- Offline PDF ingestion and chunked indexing
- Vector search with Max Marginal Relevance (MMR)
- Local LLM generation via Ollama
- Post-processing to enforce answer quality and citation structure
- Dual serving interfaces (Streamlit + FastAPI)
- Deterministic local evaluation pipeline with comparable `k` settings

Version note:
- This repository is the **v1 baseline** focused on local/offline operation and a narrow, high-signal policy corpus.

## Architecture

```mermaid
flowchart TB

  %% STYLE
  classDef data fill:#E8F1FF,stroke:#2F6FEB,stroke-width:1px,color:#0B1F44;
  classDef process fill:#F4F7FA,stroke:#7A8AA0,stroke-width:1px,color:#0F172A;
  classDef model fill:#FFF4E5,stroke:#D97706,stroke-width:1px,color:#4A2A00;
  classDef output fill:#EAFBF0,stroke:#15803D,stroke-width:1px,color:#123B1E;

  %% 1) OFFLINE
  subgraph S1["1. Offline Ingestion and Indexing"]
    direction TB
    A["Raw PDFs<br/>data/raw"]:::data
    B["Load + Clean + Metadata<br/>app/ingest.py"]:::process
    C["Chunk Documents<br/>RecursiveCharacterTextSplitter"]:::process
    D["Embed Chunks<br/>all-MiniLM-L6-v2"]:::model
    E["Persist Vector Index<br/>data/chroma"]:::data

    A --> B --> C --> D --> E
  end

  %% 2) ONLINE
  subgraph S2["2. Online Retrieval and Answering"]
    direction TB
    U["User Question"]:::data
    EP["Entry Points<br/>Streamlit (app/ui.py)<br/>FastAPI /ask (app/api.py)"]:::process
    ORCH["RAG Orchestrator<br/>ask() in app/rag.py"]:::process
    K["Top-k Selection<br/>if audit_mode=True => k=10<br/>else keep requested k"]:::process
    RET["Retrieve Context<br/>MMR Search over Chroma"]:::process
    PROMPT["Prompt Assembly<br/>SYSTEM + numbered chunks"]:::process
    LLM["Generate Answer<br/>ChatOllama"]:::model
    POST["Post-process<br/>app/postprocess.py<br/>filter/normalize citations"]:::process
    OUT["Final Response<br/>answer + citations + retrieved"]:::output

    U --> EP --> ORCH --> K --> RET --> PROMPT --> LLM --> POST --> OUT
  end

  %% 3) EVAL
  subgraph S3["3. Local Evaluation"]
    direction TB
    Q["Question Set<br/>eval/questions.json"]:::data
    EV["run_local_eval.py"]:::process
    P1["Pass 1: Retrieve Contexts"]:::process
    IDF["Build IDF Baseline"]:::process
    P2["Pass 2: Generate via ask()"]:::process
    M["Compute Metrics<br/>citation + format + overlap"]:::process
    R["Reports<br/>eval/local_eval_metrics_k*.csv/.md"]:::output

    Q --> EV --> P1 --> IDF --> P2 --> M --> R
  end

  %% CROSS-STAGE LINKS
  E --> RET
  EV --> ORCH

```

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 55, "rankSpacing": 85}}}%%
flowchart TB

  %% STYLE
  classDef data fill:#E8F1FF,stroke:#2F6FEB,stroke-width:1px,color:#0B1F44;
  classDef process fill:#F4F7FA,stroke:#7A8AA0,stroke-width:1px,color:#0F172A;
  classDef model fill:#FFF4E5,stroke:#D97706,stroke-width:1px,color:#4A2A00;
  classDef output fill:#EAFBF0,stroke:#15803D,stroke-width:1px,color:#123B1E;

  linkStyle default stroke:#64748B,stroke-width:1.2px;

  subgraph S1["1. Offline Ingestion and Indexing"]
    direction LR
    A["Raw PDFs<br/>data/raw"]:::data --> B["Load + Clean + Metadata<br/>app/ingest.py"]:::process --> C["Chunk Documents<br/>RecursiveCharacterTextSplitter"]:::process --> D["Embed Chunks<br/>all-MiniLM-L6-v2"]:::model --> E["Persist Vector Index<br/>data/chroma"]:::data
  end

  subgraph S2["2. Online Retrieval and Answering"]
    direction LR
    U["User Question"]:::data --> EP["Entry Points<br/>Streamlit (app/ui.py)<br/>FastAPI /ask (app/api.py)"]:::process --> ORCH["RAG Orchestrator<br/>ask() in app/rag.py"]:::process --> K["Top-k Selection<br/>if audit_mode=True => k=10<br/>else keep requested k"]:::process --> RET["Retrieve Context<br/>MMR Search over Chroma"]:::process --> PROMPT["Prompt Assembly<br/>SYSTEM + numbered chunks"]:::process --> LLM["Generate Answer<br/>ChatOllama"]:::model --> POST["Post-process<br/>app/postprocess.py<br/>filter/normalize citations"]:::process --> OUT["Final Response<br/>answer + citations + retrieved"]:::output
  end

  subgraph S3["3. Local Evaluation"]
    direction LR
    Q["Question Set<br/>eval/questions.json"]:::data --> EV["run_local_eval.py"]:::process --> P1["Pass 1: Retrieve Contexts"]:::process --> IDF["Build IDF Baseline"]:::process --> P2["Pass 2: Generate via ask()"]:::process --> M["Compute Metrics<br/>citation + format + overlap"]:::process --> R["Reports<br/>eval/local_eval_metrics_k*.csv/.md"]:::output
  end

  %% CROSS-STAGE LINKS (dashed, less intrusive)
  E -.-> RET
  EV -.-> ORCH

```


Detailed architecture notes: `docs/architecture.md`.

## Repository Structure

- `app/ingest.py`: PDF loading, cleaning, metadata normalization, chunking, vector DB build
- `app/rag.py`: retrieval + prompt orchestration + model call
- `app/postprocess.py`: citation parsing, answer filtering, and output normalization
- `app/ui.py`: Streamlit chat interface
- `app/api.py`: FastAPI backend (`/health`, `/ask`)
- `eval/run_local_eval.py`: deterministic local evaluation pipeline
- `eval/questions.json`: evaluation question set
- `scripts/`: convenience launch/testing scripts
- `data/raw/`: source PDFs
- `data/chroma/`: persisted vector index
- `tests/test_rag_normalization.py`: unit tests for post-processing logic

## Tech Stack

- Python, FastAPI, Streamlit
- LangChain, ChromaDB
- Ollama (`llama3.1:8b` default)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Pandas + local metric scripts for deterministic evaluation

## Quickstart

1. Create and activate environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Pull and run local model:

```bash
ollama pull llama3.1:8b
ollama serve
```

4. Build vector store:

```bash
./scripts/fetch_data.sh
python app/ingest.py
```

5. Launch app:

```bash
./scripts/run_ui.sh
```

## Running the Project

Streamlit UI:

```bash
./scripts/run_ui.sh
```

FastAPI backend:

```bash
./scripts/run_api.sh
```

CLI smoke test:

```bash
./scripts/run_cli.sh
```

API request example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "From OWASP Top 10 for LLM Applications 2025, give one mitigation idea for prompt injection.",
    "k": 8,
    "model": "llama3.1:8b",
    "audit_mode": true
  }'
```

## Data Setup

The project keeps raw source PDFs out of Git history and fetches them from official sources.

Fetch data automatically:

```bash
./scripts/fetch_data.sh
```

This script downloads into `data/raw/`:
- `NIST.CSWP.29.pdf`
- `NIST.SP.800-61r3.pdf`
- `NIST.SP.1299.pdf`
- `OWASP-Top-10-for-LLMs-v2025.pdf`

If a download URL changes upstream, the script will tell you which file to place manually.

## Example Output Format

```text
Answer:
- Constrain model behavior by providing specific instructions about the model's role, capabilities, and limitations within the system prompt, enforcing strict context adherence, limiting responses to specific tasks or topics, and instructing the model to ignore attempts to modify core instructions [3].
- Implement input and output filtering by defining sensitive categories and constructing rules for identifying and handling such content [3].
- Integrate Data Sanitization Techniques to prevent user data from entering the training model, including scrubbing or masking sensitive content before it is used in training [6].
- Enforce Strict Access Controls by limiting access to sensitive data based on the principle of least privilege, only granting access to data that is necessary for the specific user or process [6].

Cited sources:
[3] OWASP-Top-10-for-LLMs-v2025.pdf p.8
[6] OWASP-Top-10-for-LLMs-v2025.pdf p.12
```

## Visual Demo

Answer example (`k=8`):

![Answer k8](images/answer%20k8.png)

Answer example (`k=10`):

![Answer k10](images/answer%20k10.png)

Retrieved context panel:

![Retrieved context k8](images/retrieved%20context%20k8.png)

## Evaluation

Run deterministic local evaluation:

```bash
python3 eval/run_local_eval.py --top-k 8 --model llama3.1:8b
python3 eval/run_local_eval.py --top-k 10 --model llama3.1:8b
```

Latest results (from `eval/local_eval_metrics_k8.md` and `eval/local_eval_metrics_k10.md`):

- `k=8`
  - Avg citation coverage: `66.67%`
  - Avg weighted overlap: `0.6300`
  - Avg overlap: `0.6437`
  - Avg bullet-end citation rate: `66.67%`
- `k=10`
  - Avg citation coverage: `73.33%`
  - Avg weighted overlap: `0.7204`
  - Avg overlap: `0.7296`
  - Avg bullet-end citation rate: `73.33%`

Interpretation:
- `k=10` performs better than `k=8` on grounding proxies and citation-format compliance in this benchmark.
- `k=8` remains a useful lower-context option if you prioritize brevity/latency.

Evaluation summaries:

`k=8` summary:

![Eval summary k8](images/eval%20summary%20k8.png)

`k=10` summary:

![Eval summary k10](images/eval%20summary%20k10.png)

## Testing

Run unit tests:

```bash
python3 -m unittest -q tests/test_rag_normalization.py
```

Test coverage currently includes:
- citation parsing variants (`[1]`, `[1][2]`, `[1,2]`)
- filtering low-value source-list bullets
- max-5 bullet enforcement
- fallback when no valid answer bullets remain

## Known Limitations

- Retrieval-grounded generation is not guaranteed factual truth.
- Overlap metrics are proxies and can overestimate semantic correctness.
- Quality depends on local model behavior and hardware.
- OCR/PDF extraction quality directly affects retrieval quality.
- Corpus scope is intentionally narrow; broader domains may require different chunking/retrieval settings.

## Roadmap

- Add a small gold-answer evaluation set (fact + citation accuracy).
- Add optional hosted-model mode for public demo deployment.
- Migrate to newer `langchain_chroma`/`langchain_huggingface` integrations.
- Add CI to run tests and lint checks automatically.

## Future Versions (Planned)

- Add hosted LLM support, including **OpenAI models**, while preserving local Ollama mode.
- Expand corpus with additional standards, regulations, and organization-specific policy documents.
- Add ingestion metadata/versioning (source date, doc revision, chunk lineage).
- Improve retrieval with hybrid search and reranking.
- Strengthen evaluation with gold references, citation precision/recall, and regression tracking over time.
- Add a public demo deployment path (hosted API + UI mode) in addition to local-only mode.
- Add observability and guardrails (structured logs, failure analytics, stricter output schema validation).

## License

MIT License. See `LICENSE`.
