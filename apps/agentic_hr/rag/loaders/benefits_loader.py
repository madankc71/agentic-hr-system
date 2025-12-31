from pathlib import Path

BENEFITS_DIR = Path("data/benefits")

def load_benefit_documents() -> list[dict]:
    if not BENEFITS_DIR.exists():
        raise FileNotFoundError(f"Benefits folder not found: {BENEFITS_DIR}")

    docs = []

    for file in BENEFITS_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8")

        docs.append({
            "filename": file.name,
            "text": text
        })

    return docs
