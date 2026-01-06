from pathlib import Path
import faiss
import numpy as np

from apps.agentic_hr.rag.loaders.policy_loader import load_policy_documents
from apps.agentic_hr.rag.chunking.chunker import simple_chunk

from sentence_transformers import SentenceTransformer


INDEX_DIR = Path(__file__).parent / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def build_policy_index():
    print("📂 Loading policies...")
    docs = load_policy_documents()

    texts = []
    metadata = []

    for doc in docs:
        chunks = simple_chunk(doc["text"])

        for chunk in chunks:
            texts.append(chunk)
            metadata.append({"filename": doc["filename"]})

    print(f"✂️ Created {len(texts)} chunks")

    print("🧠 Embedding chunks...")
    embeddings = model.encode(texts)

    print("📦 Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Save index + metadata
    faiss.write_index(index, str(INDEX_DIR / "policy.index"))

    np.save(INDEX_DIR / "policy_texts.npy", np.array(texts, dtype=object))
    np.save(INDEX_DIR / "policy_meta.npy", np.array(metadata, dtype=object))

    print("✅ Policy index built successfully!")


if __name__ == "__main__":
    build_policy_index()
