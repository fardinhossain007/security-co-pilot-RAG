# eval/run_eval.py
from __future__ import annotations

print("✅ run_eval.py started")

import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset

from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

# Ensure local project imports win over similarly named site-packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import load_vectorstore, retrieve

QUESTIONS_PATH = Path("eval/questions.json")
OUT_DIR = Path("eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_questions():
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {QUESTIONS_PATH}. Create it first.")
    with open(QUESTIONS_PATH, "r") as f:
        qs = json.load(f)
    if not isinstance(qs, list) or not qs:
        raise ValueError("questions.json must be a non-empty list.")
    for item in qs:
        if "id" not in item or "question" not in item:
            raise ValueError("Each item must have 'id' and 'question'.")
    return qs


def generate_answers(model_name: str = "llama3.1:8b", k: int = 6):
    print("Loading vector DB...")
    db = load_vectorstore()
    print("Creating Ollama LLM...")
    llm = ChatOllama(model=model_name, temperature=0.1)

    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    rows = []
    for i, q in enumerate(questions, start=1):
        qid = q["id"]
        question = q["question"]
        print(f"[{i}/{len(questions)}] {qid}: retrieving...")
        chunks = retrieve(db, question, k=k)
        contexts = [c.text for c in chunks]

        context_block = "\n\n".join([f"[{j+1}] {contexts[j]}" for j in range(len(contexts))])
        prompt = f"""You are a security policy assistant.
Answer ONLY using the provided context. If the context does not contain the answer, say:
"I don't know based on the provided documents."

Question: {question}

Context:
{context_block}

Answer:"""

        print(f"[{i}/{len(questions)}] {qid}: generating...")
        answer = llm.invoke(prompt).content
        print(f"[{i}/{len(questions)}] {qid}: done ({len(answer)} chars)")

        rows.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
        )

    return rows


def main():
    evaluator_model = "llama3.1:8b"
    top_k = 6

    print(f"Running RAG for {QUESTIONS_PATH} using model={evaluator_model}, k={top_k} ...")
    rows = generate_answers(model_name=evaluator_model, k=top_k)

    data = Dataset.from_list(rows)

    evaluator_llm = ChatOllama(model=evaluator_model, temperature=0.0)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    print("Scoring with RAGAS (reference-free): faithfulness, answer_relevancy ...")

    faith = Faithfulness(llm=evaluator_llm)
    relev = AnswerRelevancy(llm=evaluator_llm)

    results = evaluate(
        data,
        metrics=[faith, relev],
        llm=evaluator_llm,  # keep this too; fine for some versions
        embeddings=embeddings,
    )

    df_scores = results.to_pandas()
    df_rows = pd.DataFrame(rows)
    df_full = pd.concat([df_rows, df_scores.drop(columns=["id"], errors="ignore")], axis=1)

    csv_path = OUT_DIR / "ragas_report.csv"
    md_path = OUT_DIR / "ragas_report.md"

    df_full.to_csv(csv_path, index=False)

    cols = [c for c in df_full.columns if c in ["id", "question", "faithfulness", "answer_relevancy"]]
    md_table = df_full[cols].to_markdown(index=False)
    summary = df_full[["faithfulness", "answer_relevancy"]].mean(numeric_only=True).to_dict()

    with open(md_path, "w") as f:
        f.write("# RAGAS Evaluation Report (Reference-Free)\n\n")
        f.write(f"- Evaluator model: `{evaluator_model}`\n")
        f.write(f"- Retrieval top-k: `{top_k}`\n\n")
        f.write("## Average scores\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v:.4f}\n")
        f.write("\n## Per-question scores\n\n")
        f.write(md_table)
        f.write("\n")

    print(f"✅ Saved: {csv_path}")
    print(f"✅ Saved: {md_path}")
    print("Average scores:", summary)


if __name__ == "__main__":
    main()
