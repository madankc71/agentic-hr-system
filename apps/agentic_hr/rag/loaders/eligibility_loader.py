from pathlib import Path


DATA_DIR = Path("data/eligibility")


def load_eligibility_documents() -> list[dict]:
    """
    Loads all eligibility policy files.

    Returns:
    [
        {"filename": "...", "text": "..."},
        ...
    ]
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Eligibility data folder not found: {DATA_DIR}")

    documents = []

    for file in DATA_DIR.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            documents.append(
                {
                    "filename": file.name,
                    "text": f.read(),
                }
            )

    return documents
