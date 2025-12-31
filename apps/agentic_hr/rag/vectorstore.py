from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_PATH = Path("apps/agentic_hr/rag/indexes")

INDEX_FILE = BASE_PATH / "policy.index"
TEXT_FILE = BASE_PATH / "policy_texts.npy"
META_FILE = BASE_PATH / "policy_meta.npy"

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_vectorstore():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(
            "Policy index has not been built yet. Run build_policy_index.py first."
        )

    index = faiss.read_index(str(INDEX_FILE))
    texts = np.load(TEXT_FILE, allow_pickle=True)
    metadata = np.load(META_FILE, allow_pickle=True)

    return index, texts, metadata


def search_policies(query: str, top_k: int = 3):
    index, texts, metadata = load_vectorstore()

    query_vector = model.encode([query]).astype("float32")

    scores, indices = index.search(query_vector, top_k)

    results = []

    for idx in indices[0]:
        if idx == -1:
            continue

        results.append(
            {
                "text": texts[idx],
                "meta": metadata[idx].item() if hasattr(metadata[idx], "item") else metadata[idx],
            }
        )

    return results

def search_benefits(query: str, top_k: int = 3):
    index_file = BASE_PATH / "benefits.index"
    meta_file = BASE_PATH / "benefits_meta.npy"

    if not index_file.exists() or not meta_file.exists():
        raise FileNotFoundError("Benefits index not built yet.")

    index = faiss.read_index(str(index_file))
    metadata = np.load(meta_file, allow_pickle=True)

    query_vec = model.encode([query]).astype("float32")
    scores, indices = index.search(query_vec, top_k)

    results = []

    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(metadata[idx])

    return results
