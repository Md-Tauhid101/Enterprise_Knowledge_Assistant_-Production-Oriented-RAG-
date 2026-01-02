# schema.py
import psycopg2
from config import DB_CONFIG


def run_schema():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        ### documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id UUID PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            checksum TEXT NOT NULL UNIQUE,
            version INTEGER NOT NULL DEFAULT 1,
            ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """)

        ### chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id UUID PRIMARY KEY,
                document_id UUID NOT NULL
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER,
                raw_text TEXT NOT NULL,
                clean_text TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (document_id, chunk_hash)
            );
        """)


        ### INDEXES
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks(document_id);
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_hash
            ON chunks(chunk_hash);
        """)

        conn.commit()
        print("✅ Schema created successfully")
    except Exception as e:
        conn.rollback()
        print("❌ Shcema creation failed.")
        raise e
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_schema()