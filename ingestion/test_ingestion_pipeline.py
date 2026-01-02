# ingestion_pipeline.py

from typing import List
from ingestion.load import load_documents
from ingestion.clean import clean_text
from ingestion.chunks import chunk_documents
from langchain_core.documents import Document


def run_ingestion(file_paths: List[str]) -> List[Document]:
    print("=== INGESTION PIPELINE STARTED ===")

    # ---------------------------
    # STEP 1: LOAD DOCUMENTS
    # ---------------------------
    print("\n[STEP 1] Loading documents...")
    docs = load_documents(file_paths)
    print(f"Loaded {len(docs)} elements")

    if not docs:
        print("No documents loaded. Exiting pipeline.")
        return []

    # ---------------------------
    # STEP 2: CLEAN TEXT
    # ---------------------------
    print("\n[STEP 2] Cleaning text elements...")
    cleaned_docs = []

    for doc in docs:
        if doc.metadata.get("element_type") == "Image":
            cleaned_docs.append(doc)
            continue

        cleaned_text = clean_text(doc.page_content)
        if not cleaned_text:
            continue

        cleaned_docs.append(
            Document(
                page_content=cleaned_text,
                metadata=doc.metadata,
            )
        )

    print(f"Cleaned elements count: {len(cleaned_docs)}")

    # ---------------------------
    # STEP 3: CHUNK DOCUMENTS
    # ---------------------------
    print("\n[STEP 3] Chunking documents...")
    chunks = chunk_documents(cleaned_docs)
    print(f"Generated {len(chunks)} chunks")

    # ---------------------------
    # SAMPLE OUTPUT
    # ---------------------------
    print("\n[SAMPLE CHUNKS]")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1}")
        print("-" * 40)
        print(f"chunk_id   : {chunk.metadata.get('chunk_id')}")
        print(f"element_id : {chunk.metadata.get('element_id')}")
        print(f"doc_id     : {chunk.metadata.get('doc_id')}")
        print(f"text (100 chars): {chunk.page_content[:100]}")

    print("\n=== INGESTION PIPELINE COMPLETED ===")
    return chunks


if __name__ == "__main__":
    FILES_TO_INGEST = ["./data/India Post.pdf", "./data/instagram data.csv", "./data/ppt.pptx", "./data/Tauhid_CV.pdf"]

    run_ingestion(FILES_TO_INGEST)
