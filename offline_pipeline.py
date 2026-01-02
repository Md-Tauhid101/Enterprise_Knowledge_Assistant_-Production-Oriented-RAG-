# offline_pipeline.py

from typing import List
from PIL import Image
from langchain_core.documents import Document

from config import DB_CONFIG

from ingestion.load import load_documents
from ingestion.chunks import chunk_documents

from storage.postgres import PostgresStore
from storage.multimodel_vector_store import MultiModalVectorStore

from indexes.dense_embeddings import embed_text, embed_image, embed_table

VERSION = 1
FILES_TO_INGEST = ["./data/Tauhid_CV.pdf"]
# FILES_TO_INGEST = ["./data/India Post.pdf", "./data/instagram data.csv", "./data/ppt.pptx", "./data/Tauhid_CV.pdf", "./data/tiger.jpg"]

# OFFLINE PIPELINE
def run_offline_pipeline(files: List[str]) -> None:
    print("\n=== OFFLINE INGESTION PIPELINE STARTED ===\n")
    
    # Load raw documents
    print("[1] Loading documents...")
    raw_docs = load_documents(files)
    print(f"    Loaded {len(raw_docs)} raw elements")

    # Chunk documents
    print("[2] Chunking documents...")
    chunks = chunk_documents(raw_docs)
    print(f"    Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError("Chunking produced zero chunks")

    # NOTE: for now single source file per ingestion run
    first_doc = raw_docs[0]

    source_path = first_doc.metadata.get("source_path")
    source_type = first_doc.metadata.get("source_type")
    checksum = first_doc.metadata.get("checksum")

    # Store chunks in Postgres
    print("[3] Storing chunks in Postgres...")
    pg = PostgresStore(DB_CONFIG)
    pg.insert_document_with_chunks(
        source_path=source_path,
        source_type=source_type,
        checksum=checksum,
        chunks=chunks,
        version=VERSION
    )
    print("✅ Chunks stored successfully")


    # Initialize vector stores
    print("[4] Initializing vector stores...")
    mm_store = MultiModalVectorStore()
    mm_store.reset_all()
    print("Vector stores reset")

    # Create embeddings + store
    print("[5] Creating embeddings and storing vectors...")

    text_count = table_count = image_count = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        # ---- TEXT ----
        if chunk.get("clean_text"):
            emb = embed_text(chunk["clean_text"])
            mm_store.add_text(emb, chunk_id)
            text_count += 1

        # ---- TABLE ----
        if chunk.get("table_text"):
            emb = embed_table(chunk["table_text"])
            if emb is not None:
                mm_store.add_table(emb, chunk_id)
                table_count += 1

        # ---- IMAGE ----
        if chunk.get("image_path"):
            try:
                img = Image.open(chunk["image_path"])
                emb = embed_image(img)
                mm_store.add_image(emb, chunk_id)
                image_count += 1
            except Exception as e:
                print(f"    ⚠️ Failed image embedding for {chunk_id}: {e}")

    print(f"    Text embeddings:  {text_count}")
    print(f"    Table embeddings: {table_count}")
    print(f"    Image embeddings: {image_count}")

    # Persist vector stores
    print("[7] Saving vector indexes to disk...")
    mm_store.save_all()
    print("    Vector stores saved")

    print("\n=== OFFLINE INGESTION PIPELINE COMPLETED ===\n")

if __name__ == "__main__":
    run_offline_pipeline(FILES_TO_INGEST)