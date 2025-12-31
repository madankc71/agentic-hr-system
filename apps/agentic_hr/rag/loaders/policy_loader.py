import os
from pathlib import Path


DATA_DIR = Path("data/employment_policies")


def load_policy_documents() -> list[dict]:
    """
    Loads all policy text files from the dataset folder.

    Returns a list like:
    [
        {"filename": "leave_policy.txt", "text": "..."},
        ...
    ]
    """

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Policy data folder not found: {DATA_DIR}")

    documents = []

    for file in DATA_DIR.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append(
            {
                "filename": file.name,
                "text": text,
            }
        )

    return documents