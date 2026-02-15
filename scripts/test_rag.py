from app.rag import ask

questions = [
    "What are the CSF 2.0 Functions?",
    "According to NIST SP 800-61r3, what is incident response?",
    "What is one key risk from the OWASP Top 10 for LLM Applications 2025 related to prompt injection?",
]

for q in questions:
    print("=" * 100)
    print("Q:", q)
    out = ask(q, k=6)
    print(out["answer"])
