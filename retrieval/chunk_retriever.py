# storage/chunk_retriever.py
import psycopg2
from psycopg2.extras import RealDictCursor


class ChunkRetriever:
    """
    Read-only access layer for chunk storage.

    Responsibility:
    - Fetch ground-truth chunk content by chunk_id
    - Nothing else
    """

    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)

    def get_chunk(self, chunk_id: str) -> dict:
        """
        Fetch a single chunk by chunk_id.

        Raises:
            ValueError if chunk does not exist
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    chunk_index,
                    page_number,
                    clean_text,
                    created_at
                FROM chunks
                WHERE chunk_id = %s
                """,
                (chunk_id,)
            )

            row = cur.fetchone()

        if row is None:
            raise ValueError(f"Chunk not found: {chunk_id}")

        return {
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "clean_text": row["clean_text"],
            "created_at": row["created_at"],
        }

if __name__ == "__main__":
    from config import DB_CONFIG
    retrieve = ChunkRetriever(DB_CONFIG)
    print(retrieve.get("5b99de2f-45ff-4703-85a4-fe9f884a7d66"))