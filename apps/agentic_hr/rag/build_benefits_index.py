from pathlib import Path
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from loaders.benefits_loader import load_benefit_documents
from chunking.chunker import simple_chunk


INDEX_DIR = Path(__file__).parent / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


def build_benefits_index():
    docs = load_benefit_documents()

    texts = []
    metadata = []

    for doc in docs:
        chunks = simple_chunk(doc["text"])

        for chunk in chunks:
            texts.append(chunk)
            metadata.append({
                "filename": doc["filename"],
                "text": chunk,
            })

    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, str(INDEX_DIR / "benefits.index"))

    np.save(INDEX_DIR / "benefits_meta.npy", np.array(metadata, dtype=object))

    print("Benefits index built successfully!")


if __name__ == "__main__":
    build_benefits_index()