from typing import List


def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits long text into overlapping chunks.

    Example:
    ABCDEF with chunk_size=4 and overlap=2
    -> ABCD, CDEF
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # move start ahead but keep overlap for context
        start += chunk_size - overlap

    return chunks