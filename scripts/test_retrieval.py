from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "security_corpus_v1"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

db = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME),
)

query = "What are the five functions of NIST CSF 2.0?"
docs = db.similarity_search(query, k=4)

print("Query:", query)
print("=" * 80)
for i, d in enumerate(docs, 1):
    meta = d.metadata
    print(f"[{i}] {meta.get('source_file')} | page {meta.get('page_number')}")
    print(d.page_content[:600])
    print("-" * 80)
