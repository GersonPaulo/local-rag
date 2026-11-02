
from pathlib import Path

import chromadb
import fitz  
#import PyMuPDF
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = Path("/home/elpapi/Documents/py_projects/zlocal-rag/documents")
DB_DIR = "chroma_db"
COLLECTION_NAME = "learning_rag"

# Local embedding model; downloaded on first use.
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as pdf:
            return "\n".join(page.get_text() for page in pdf)

    return ""

def chunk_text(text: str, chunk_size=180, overlap=40):
    # Simple word-based chunking for learning.
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks

documents = []
metadatas = []
ids = []

for path in DOCUMENTS_DIR.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() not in {".txt", ".pdf"}:
        continue

    text = extract_text(path)

    for index, chunk in enumerate(chunk_text(text)):
        documents.append(chunk)
        metadatas.append({
            "source": str(path),
            "chunk": index
        })
        ids.append(f"{path.as_posix()}-{index}")

if not documents:
    raise RuntimeError("No readable documents found.")

# Rebuild the collection on every ingestion run.
# This avoids duplicate IDs and stale chunks in this tutorial.
collection.get()["ids"]


embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
).tolist()

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print(f"Indexed {len(documents)} chunks.")
print(f"Database saved locally in: {DB_DIR}")