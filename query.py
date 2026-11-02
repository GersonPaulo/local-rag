
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION_NAME = "learning_rag"
LLM_NAME = "qwen2.5:3b"

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

def retrieve(question: str, top_k: int = 4):
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "text": document,
            "source": metadata["source"],
            "chunk": metadata["chunk"],
            "distance": distance
        }
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

def answer_question(question: str):
    chunks = retrieve(question)

    if not chunks:
        return "No relevant documents were found."

    context = "\n\n".join(
        f"[Source: {item['source']}, chunk: {item['chunk']}]\n"
        f"{item['text']}"
        for item in chunks
    )

    response = ollama.chat(
        model=LLM_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the supplied context. "
                    "If the context does not contain the answer, "
                    "say that the documents do not provide enough "
                    "information. Cite the source filenames."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        options={"temperature": 0}
    )

    return response["message"]["content"]

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.strip().lower() == "exit":
            break

        if not question.strip():
            continue

        print("\nAnswer:\n")
        print(answer_question(question))