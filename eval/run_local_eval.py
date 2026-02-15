# eval/run_eval_local.py
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Ensure local project imports win over similarly named site-packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import load_vectorstore, retrieve, ask  # uses your real RAG prompt/format
from app.postprocess import extract_bullets as extract_answer_bullets


QUESTIONS_PATH = Path("eval/questions.json")
OUT_DIR = Path("eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TOP_K = 8


# -----------------------------
# Text processing helpers
# -----------------------------
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","to","of","in","on","for","with","by","as","at",
    "is","are","was","were","be","been","being","it","this","that","these","those","from","into","over","under","up",
    "down","out","about","than","so","such","not","no","yes","can","could","may","might","should","would","will","just",
    "also","more","most","less","least","very","any","all","each","some","many","much","few","between","within","across",
    "their","there","they","them","we","you","your","our","i","he","she","his","her","its"
}

CITATION_RE = re.compile(r"\[\d+(?:\]\[\d+)*\]|\[\d+(?:,\s*\d+)*\]")
BULLET_RE = re.compile(r"^\s*-\s+")

PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def load_questions() -> List[Dict]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {QUESTIONS_PATH}. Create it first.")
    qs = json.loads(QUESTIONS_PATH.read_text())
    if not isinstance(qs, list) or not qs:
        raise ValueError("questions.json must be a non-empty list.")
    for item in qs:
        if "id" not in item or "question" not in item:
            raise ValueError("Each item must have 'id' and 'question'.")
    return qs


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def tokenize(text: str) -> List[str]:
    t = text.lower().translate(PUNCT_TABLE)
    toks = [w for w in t.split() if w and w not in STOPWORDS and len(w) > 2]
    return toks


def bullet_has_citation(bullet_line: str) -> bool:
    # require at least one [#] somewhere (ideally end, but we'll count presence)
    return bool(CITATION_RE.search(bullet_line))


def answer_has_heading(answer: str) -> int:
    return 1 if re.search(r"(?im)^\s*Answer:\s*$", answer) else 0


def answer_has_cited_sources_heading(answer: str) -> int:
    return 1 if re.search(r"(?im)^\s*Cited sources:\s*$", answer) else 0


def citation_ids_in_text(text: str) -> List[int]:
    ids = []
    for raw in CITATION_RE.findall(text):
        # Handles [1], [1,2], [1][2]
        pieces = raw.replace("][", ",").replace("[", "").replace("]", "").split(",")
        for p in pieces:
            p = p.strip()
            if p.isdigit():
                ids.append(int(p))
    return ids


def bullet_ends_with_citation(bullet_line: str) -> bool:
    # Allow [1], [1,2], [1][2] with optional trailing punctuation.
    return bool(
        re.search(
            r"(?:\[\d+(?:,\s*\d+)*\]|\[\d+\](?:\[\d+\])+)[\s\.;:,!?)]*$",
            bullet_line.strip(),
        )
    )


# -----------------------------
# IDF computation (weighted overlap)
# -----------------------------
def build_idf(contexts_per_q: List[List[str]]) -> Dict[str, float]:
    """
    Compute IDF over the evaluation set using per-question context as a "document".
    IDF(t) = log((N+1)/(df+1)) + 1
    """
    N = len(contexts_per_q)
    df: Dict[str, int] = {}

    for ctx_list in contexts_per_q:
        # combine all retrieved chunks for this question into one doc
        combined = " ".join(ctx_list)
        doc_tokens = set(tokenize(combined))
        for t in doc_tokens:
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {}
    for t, d in df.items():
        idf[t] = math.log((N + 1) / (d + 1)) + 1.0
    return idf


def overlap_score(bullet_text: str, context_text: str) -> float:
    bt = tokenize(bullet_text)
    if not bt:
        return 0.0
    ct = set(tokenize(context_text))
    inter = sum(1 for t in bt if t in ct)
    return inter / len(bt)


def weighted_overlap_score(bullet_text: str, context_text: str, idf: Dict[str, float]) -> float:
    bt = tokenize(bullet_text)
    if not bt:
        return 0.0
    ct = set(tokenize(context_text))
    weights = [idf.get(t, 1.0) for t in bt]
    denom = sum(weights) if weights else 0.0
    if denom == 0.0:
        return 0.0
    numer = sum(idf.get(t, 1.0) for t in bt if t in ct)
    return numer / denom


# -----------------------------
# Main eval
# -----------------------------
def main(top_k: int = DEFAULT_TOP_K, model: str = DEFAULT_MODEL):
    print("✅ run_eval_local.py started")
    print(f"MODEL={model} | TOP_K={top_k}")

    qs = load_questions()
    db = load_vectorstore()

    # First pass: retrieve contexts (for IDF and overlap)
    contexts_per_q: List[List[str]] = []
    for q in qs:
        chunks = retrieve(db, q["question"], k=top_k)
        contexts_per_q.append([c.text for c in chunks])

    idf = build_idf(contexts_per_q)

    rows = []
    for idx, q in enumerate(qs):
        qid = q["id"]
        question = q["question"]

        contexts = contexts_per_q[idx]
        combined_context = "\n".join(contexts)

        # Generate using your real app RAG (so eval matches UI behavior)
        out = ask(question, k=top_k, model=model, audit_mode=(top_k >= 10))
        answer = out["answer"]

        bullets = extract_answer_bullets(answer)
        n_bullets = len(bullets)

        cited_bullets = [b for b in bullets if bullet_has_citation(b)]
        citation_coverage = (len(cited_bullets) / n_bullets) if n_bullets else 0.0
        has_any_citation = 1 if len(cited_bullets) > 0 else 0

        # Format compliance metrics
        has_answer_heading = answer_has_heading(answer)
        has_cited_sources_heading = answer_has_cited_sources_heading(answer)
        bullet_count_valid = 1 if 1 <= n_bullets <= 5 else 0
        bullet_end_citation_rate = (
            sum(1 for b in bullets if bullet_ends_with_citation(b)) / n_bullets if n_bullets else 0.0
        )

        # Valid in-range citation usage over all bullet citation IDs
        all_bullet_citation_ids: List[int] = []
        for b in bullets:
            all_bullet_citation_ids.extend(citation_ids_in_text(b))

        if all_bullet_citation_ids:
            valid_ids = [i for i in all_bullet_citation_ids if 1 <= i <= top_k]
            valid_citation_id_rate = len(valid_ids) / len(all_bullet_citation_ids)
        else:
            valid_citation_id_rate = 0.0

        # Overlap metrics per bullet (strip citations first)
        overlaps = []
        w_overlaps = []
        for b in bullets:
            clean_b = strip_citations(b)
            overlaps.append(overlap_score(clean_b, combined_context))
            w_overlaps.append(weighted_overlap_score(clean_b, combined_context, idf))

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        min_overlap = min(overlaps) if overlaps else 0.0

        avg_w_overlap = sum(w_overlaps) / len(w_overlaps) if w_overlaps else 0.0
        min_w_overlap = min(w_overlaps) if w_overlaps else 0.0

        rows.append(
            {
                "id": qid,
                "question": question,
                "top_k": top_k,
                "num_bullets": n_bullets,
                "citation_coverage": citation_coverage,
                "has_any_citation": has_any_citation,
                "has_answer_heading": has_answer_heading,
                "has_cited_sources_heading": has_cited_sources_heading,
                "bullet_count_valid_1to5": bullet_count_valid,
                "bullet_end_citation_rate": bullet_end_citation_rate,
                "valid_citation_id_rate": valid_citation_id_rate,
                "avg_overlap": avg_overlap,
                "min_overlap": min_overlap,
                "avg_weighted_overlap": avg_w_overlap,
                "min_weighted_overlap": min_w_overlap,
            }
        )

        print(
            f"[{idx+1}/{len(qs)}] {qid} | bullets={n_bullets} | cite_cov={citation_coverage:.2f} "
            f"| avg_ov={avg_overlap:.2f} | avg_wov={avg_w_overlap:.2f}"
        )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"local_eval_metrics_k{top_k}.csv"
    md_path = OUT_DIR / f"local_eval_metrics_k{top_k}.md"

    df.to_csv(csv_path, index=False)

    summary = {
        "questions": len(df),
        "avg_citation_coverage": float(df["citation_coverage"].mean()) if len(df) else 0.0,
        "avg_overlap": float(df["avg_overlap"].mean()) if len(df) else 0.0,
        "avg_weighted_overlap": float(df["avg_weighted_overlap"].mean()) if len(df) else 0.0,
        "avg_num_bullets": float(df["num_bullets"].mean()) if len(df) else 0.0,
        "pct_any_citation": float(df["has_any_citation"].mean()) if len(df) else 0.0,
        "pct_has_answer_heading": float(df["has_answer_heading"].mean()) if len(df) else 0.0,
        "pct_has_cited_sources_heading": float(df["has_cited_sources_heading"].mean()) if len(df) else 0.0,
        "pct_bullet_count_valid_1to5": float(df["bullet_count_valid_1to5"].mean()) if len(df) else 0.0,
        "avg_bullet_end_citation_rate": float(df["bullet_end_citation_rate"].mean()) if len(df) else 0.0,
        "avg_valid_citation_id_rate": float(df["valid_citation_id_rate"].mean()) if len(df) else 0.0,
    }

    with open(md_path, "w") as f:
        f.write("# Local RAG Evaluation Report (Deterministic)\n\n")
        f.write(f"- Model: `{model}`\n")
        f.write(f"- Retrieval top-k: `{top_k}`\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Avg citation coverage (bullets with [x])**: {summary['avg_citation_coverage']:.2%}\n")
        f.write(f"- **% answers with any citation**: {summary['pct_any_citation']:.2%}\n")
        f.write(f"- **% answers with `Answer:` heading**: {summary['pct_has_answer_heading']:.2%}\n")
        f.write(f"- **% answers with `Cited sources:` heading**: {summary['pct_has_cited_sources_heading']:.2%}\n")
        f.write(f"- **% answers with valid bullet count (1-5)**: {summary['pct_bullet_count_valid_1to5']:.2%}\n")
        f.write(f"- **Avg bullet-end citation rate**: {summary['avg_bullet_end_citation_rate']:.2%}\n")
        f.write(f"- **Avg valid citation ID rate (within top-k)**: {summary['avg_valid_citation_id_rate']:.2%}\n")
        f.write(f"- **Avg overlap (0–1)**: {summary['avg_overlap']:.4f}\n")
        f.write(f"- **Avg weighted overlap (IDF, 0–1)**: {summary['avg_weighted_overlap']:.4f}\n")
        f.write(f"- **Avg bullets per answer**: {summary['avg_num_bullets']:.2f}\n\n")
        f.write("## Per-question metrics\n\n")
        show_cols = [
            "id",
            "num_bullets",
            "citation_coverage",
            "bullet_count_valid_1to5",
            "bullet_end_citation_rate",
            "valid_citation_id_rate",
            "avg_overlap",
            "avg_weighted_overlap",
            "min_overlap",
            "min_weighted_overlap",
        ]
        f.write(df[show_cols].to_markdown(index=False))
        f.write("\n")

    print(f"✅ Saved: {csv_path}")
    print(f"✅ Saved: {md_path}")
    print("Summary:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local deterministic RAG evaluation.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Retriever top-k (default: 8).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name.")
    args = parser.parse_args()
    main(top_k=args.top_k, model=args.model)
