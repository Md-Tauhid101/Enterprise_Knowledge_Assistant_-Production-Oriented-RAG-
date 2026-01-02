# chunks.py

from typing import List, Dict
from langchain_core.documents import Document
from ingestion.clean import clean_text
import uuid

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP = 150


def chunk_documents(
    docs: List[Document],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict]:
    """
    Convert loaded Documents into cleaned, retrieval-ready chunks.

    Responsibilities:
    - Convert Document → plain dict
    - Clean text exactly once
    - Create deterministic chunk_ids
    - Preserve full traceability
    """

    chunks: List[Dict] = []

    for doc in docs:
        meta = doc.metadata
        raw_text = doc.page_content or ""

        # ---------------------------
        # IMAGE ELEMENT (no chunking)
        # ---------------------------
        if meta.get("element_type") == "Image":
            chunk_id = f"{meta['element_id']}_0"

            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": 0,
                "raw_text": raw_text,
                "clean_text": raw_text,  # nothing to clean
                "metadata": meta,
            })
            continue

        # ---------------------------
        # TEXT ELEMENT
        # ---------------------------
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        text_len = len(cleaned)
        start = 0
        chunk_index = 0

        while start < text_len:
            end = start + max_chars
            chunk_text = cleaned[start:end].strip()

            if not chunk_text:
                break

            # chunk_id = f"{meta['element_id']}_{chunk_index}"
            chunk_id = str(uuid.uuid4())

            chunks.append({
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "raw_text": raw_text[start:end].strip(),   # original slice
                "clean_text": chunk_text,              # cleaned slice
                "metadata": meta,
            })

            chunk_index += 1
            start = end - overlap
            if start < 0:
                start = 0

    return chunks
