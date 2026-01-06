from pathlib import Path
import faiss
import numpy as np

from apps.agentic_hr.rag.loaders.eligibility_loader import load_eligibility_documents
from apps.agentic_hr.rag.chunking.chunker import simple_chunk

from sentence_transformers import SentenceTransformer


INDEX_DIR = Path(__file__).parent / "indexes"
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_eligibility_index():
    print("📂 Loading eligibility docs...")
    docs = load_eligibility_documents()

    texts = []
    metadata = []

    for doc in docs:
        chunks = simple_chunk(doc["text"])

        for chunk in chunks:
            texts.append(chunk)
            metadata.append({"filename": doc["filename"]})

    print(f"✂️ Created {len(texts)} chunks")

    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, str(INDEX_DIR / "eligibility.index"))

    np.save(INDEX_DIR / "eligibility_texts.npy", np.array(texts, dtype=object))
    np.save(INDEX_DIR / "eligibility_meta.npy", np.array(metadata, dtype=object))

    print("✅ Eligibility index built successfully!")


if __name__ == "__main__":
    build_eligibility_index()