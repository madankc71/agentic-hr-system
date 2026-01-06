# from typing import List


# def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
#     """
#     Splits long text into overlapping chunks.

#     Example:
#     ABCDEF with chunk_size=4 and overlap=2
#     -> ABCD, CDEF
#     """

#     chunks = []
#     start = 0

#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)

#         # move start ahead but keep overlap for context
#         start += chunk_size - overlap

#     return chunks

from typing import List


def simple_chunk(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    """
    Hybrid chunking:
    1) split by paragraphs
    2) merge until size limit
    3) add small overlap for context continuity
    """

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        # If adding paragraph stays below limit — keep combining
        if len(current) + len(p) < max_chars:
            current += ("\n" + p) if current else p
        else:
            chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    # ---- ADD OVERLAP ----
    final_chunks = []
    for i, ch in enumerate(chunks):
        if i == 0:
            final_chunks.append(ch)
            continue

        prev = chunks[i - 1]
        overlap_text = prev[-overlap:]

        final_chunks.append(overlap_text + "\n" + ch)

    return final_chunks
