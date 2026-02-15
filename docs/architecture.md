# Architecture

## End-to-End Flow

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

## Components

- Ingestion: extracts page text from PDFs, normalizes text, assigns metadata (`source_file`, `page_number`), then chunks for indexing.
- Retrieval: uses MMR to balance relevance and diversity across chunks.
- Generation: prompts local Ollama model to answer using only retrieved context.
- Post-processing: enforces answer shape and citation rules, removes low-value source-list outputs, and keeps only cited chunk mappings.
- Evaluation: local scripts run two-pass scoring (retrieve + generate), compute citation/format compliance, and score lexical grounding proxies (including IDF-weighted overlap).

## Design Choices

- Local-first stack (Ollama + Chroma) keeps costs near zero.
- Explicit chunk numbering enables controllable citation references in output.
- Post-generation normalization improves robustness against prompt-format failures.
