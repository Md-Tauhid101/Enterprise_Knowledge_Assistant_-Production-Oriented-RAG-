# sparse_index.py

from typing import List, Dict
from rank_bm25 import BM25Okapi
import re
from config import DB_CONFIG
from storage.postgres import PostgresStore
import pickle


class BM25Index:
    """
    Sparse retrieval using BM25.
    Used ONLY for candidate retrieval.
    Never treated as ground truth.
    """

    def __init__(self):
        self.bm25 = None
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []

    # Simple Tokenization
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        # keep words + numbers, drop punctuation
        return re.findall(r"\b\w+\b", text)

    # Build index
    def build(self, chunks: List[Dict[str, str]]) -> None:
        """
        chunks: List of dicts with keys:
            - chunk_id: str
            - text: str (cleaned_text)
        """

        self.chunk_ids = []
        self.corpus_tokens = []

        for chunk in chunks:
            text = chunk.get("clean_text")
            chunk_id = chunk.get("chunk_id")

            if not text or not chunk_id:
                continue

            tokens = self._tokenize(text)

            # skip empty / junk chunks
            if not tokens:
                continue

            self.chunk_ids.append(chunk_id)
            self.corpus_tokens.append(tokens)

        if not self.corpus_tokens:
            raise ValueError("BM25 index build failed: empty corpus")

        self.bm25 = BM25Okapi(self.corpus_tokens)

    # Save index
    def save(self, path: str) -> None:
        if self.bm25 is None:
            raise RuntimeError("Cannot save BM25 index: index not built")

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "corpus_tokens": self.corpus_tokens,
                    "bm25": self.bm25,
                },
                f
            )
    # Load index
    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.chunk_ids = data["chunk_ids"]
        self.corpus_tokens = data["corpus_tokens"]
        self.bm25 = data["bm25"]


    # Search
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Returns:
        [
            {
                "chunk_id": str,
                "score": float
            }
        ]
        """

        if self.bm25 is None:
            raise RuntimeError("BM25 index not built")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # rank by score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "score": float(score)
            })
        # print(f"✅✅ SUCCESSFULLY SEARCHED SPARSE EMBEDDINGS: {len(results)}")
        return results

if __name__ == "__main__":
    from indexes.sparse_index import BM25Index
    from storage.postgres import PostgresStore
    from config import DB_CONFIG

    pg = PostgresStore(DB_CONFIG)

    BM25_INDEX_PATH = "sparse_store/bm25_index.pkl"

    print("[BM25] Loading chunks from database...")

    chunks = pg.fetch_all_chunks()
    if not chunks:
        raise RuntimeError("No chunks found. Cannot build BM25 index.")

    print(f"[BM25] Loaded {len(chunks)} chunks")

    bm25 = BM25Index()

    print("[BM25] Building sparse index...")
    bm25.build(chunks)

    print("[BM25] Saving index to disk...")
    bm25.save(BM25_INDEX_PATH)

    print(f"[BM25] Index saved at: {BM25_INDEX_PATH}")