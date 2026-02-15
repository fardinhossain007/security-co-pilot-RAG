from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field
from app.rag import ask

app = FastAPI(title="Security Policy Copilot (Local RAG)")

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    k: int = Field(default=8, ge=3, le=12)
    model: str = Field(default="llama3.1:8b", min_length=1, max_length=100)
    audit_mode: bool = False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_endpoint(req: AskRequest):
    try:
        return ask(req.question, k=req.k, model=req.model, audit_mode=req.audit_mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG request failed: {exc}") from exc
