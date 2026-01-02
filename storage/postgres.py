# postgres.py
import psycopg2
import uuid
import hashlib
from psycopg2.extras import execute_batch
from psycopg2.extras import RealDictCursor


class PostgresStore:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.conn.autocommit = False  # IMPORTANT

    # ----------------------------
    # DOCUMENT + CHUNKS (ATOMIC)
    # ----------------------------
    def insert_document_with_chunks(
        self,
        source_path,
        source_type,
        checksum,
        chunks,
        version=1
    ):
        document_id = uuid.uuid4()

        try:
            with self.conn.cursor() as cur:
                # insert document
                cur.execute("""
                    INSERT INTO documents (
                        document_id, source_path, source_type, checksum, version
                    )
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    str(document_id),
                    source_path,
                    source_type,
                    checksum,
                    version
                ))

                # prepare chunk rows
                rows = []
                for idx, chunk in enumerate(chunks):
                    # chunk_id = uuid.uuid4()
                    chunk_id = chunk["chunk_id"]
                    chunk_hash = hashlib.sha256(
                        chunk["clean_text"].encode("utf-8")
                    ).hexdigest()

                    rows.append((
                        str(chunk_id),
                        str(document_id),
                        idx,
                        chunk["raw_text"],
                        chunk["clean_text"],
                        chunk_hash
                    ))

                execute_batch(
                    cur,
                    """
                    INSERT INTO chunks (
                        chunk_id,
                        document_id,
                        chunk_index,
                        raw_text,
                        clean_text,
                        chunk_hash
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    rows
                )

            self.conn.commit()
            return document_id

        except Exception:
            self.conn.rollback()
            raise

    def fetch_all_chunks(self):
        """
        Fetch all chunks for building retrieval indexes.
        Returns:
        [
            {
                "chunk_id": str,
                "clean_text": str
            }
        ]
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        chunk_id,
                        clean_text
                    FROM chunks
                """)
                rows = cur.fetchall()
        finally:
            self.conn.close()

        if not rows:
            raise ValueError("No chunks found in database")

        return [
            {
                "chunk_id": str(r["chunk_id"]),
                "clean_text": r["clean_text"]
            }
            for r in rows
        ]

    def close(self):
        self.conn.close()
