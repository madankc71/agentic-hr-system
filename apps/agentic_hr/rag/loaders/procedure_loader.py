from pathlib import Path


DATA_DIR = Path("data/hr_procedures")


def load_procedure_documents() -> list[dict]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing dataset folder: {DATA_DIR}")

    docs = []

    for file in DATA_DIR.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append({
                "filename": file.name,
                "text": f.read()
            })

    return docs