# vector_store.py
import os
import json
import faiss
import numpy as np
from typing import List, Dict

# class VectorStore:
#     """
#     Persistent dense vector store using FAISS.

#     Responsibilities:
#     - Store vectors
#     - Maintain id mapping
#     - Perform similarity search
#     - Save / load from disk
#     - Reset index on re-ingestion
#     """

#     def __init__(self, dim: int, base_path: str = "./vector_store"):
#         self.dim = dim
#         self.base_path = base_path
#         os.makedirs(base_path, exist_ok=True)

#         self.index_path = os.path.join(base_path, "vectors.index")
#         self.id_map_path = os.path.join(base_path, "id_map.json")

#         if os.path.exists(self.index_path):
#             self.index = faiss.read_index(self.index_path)
#             with open(self.id_map_path, "r") as f:
#                 self.id_map = json.load(f)
#             self.id_map = {int(k): v for k, v in self.id_map.items()}
#         else:
#             self.index = faiss.IndexFlatIP(dim)
#             self.id_map: Dict[int, str] = {}

#     # RESET (used during re-ingestion)
#     def reset(self) -> None:
#         """
#         Completely reset the vector store.
#         Deletes existing index and ID map from disk and memory.
#         """

#         # delete on-disk files if they exist
#         if os.path.exists(self.index_path):
#             os.remove(self.index_path)

#         if os.path.exists(self.id_map_path):
#             os.remove(self.id_map_path)

#         # reinitialize empty index and mapping
#         self.index = faiss.IndexFlatIP(self.dim)
#         self.id_map = {}

#     # Add vectors
#     def add(self, vector: np.ndarray, chunk_id: str) -> None:
#         if vector.ndim != 1:
#             raise ValueError("Vector must be 1D")

#         if vector.shape[0] != self.dim:
#             raise ValueError(
#                 f"Vector dim mismatch: expected {self.dim}, got {vector.shape[0]}"
#             )

#         vector = vector.astype("float32").reshape(1, -1)

#         idx = self.index.ntotal
#         self.index.add(vector)
#         self.id_map[idx] = chunk_id

#     # Search
#     def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
#         if self.index.ntotal == 0:
#             return []

#         if query_vector.ndim != 1:
#             raise ValueError("Query vector must be 1D")

#         query_vector = query_vector.astype("float32").reshape(1, -1)

#         scores, indices = self.index.search(query_vector, top_k)

#         results = []
#         for idx, score in zip(indices[0], scores[0]):
#             if idx == -1:
#                 continue
#             results.append({
#                 "chunk_id": self.id_map[idx],
#                 "score": float(score)
#             })

#         return results
    
#     # Persistence
#     def save(self) -> None:
#         faiss.write_index(self.index, self.index_path)
#         with open(self.id_map_path, "w") as f:
#             json.dump(self.id_map, f)

#     def size(self) -> int:
#         return self.index.ntotal

class VectorStore:
    """
    Persistent dense vector store using FAISS (Inner Product / Cosine).

    Guarantees:
    - Index is always usable after construction
    - Dimensional consistency enforced
    - Safe disk loading
    - Defensive normalization
    """

    def __init__(self, dim: int, base_path: str = "./vector_store"):
        self.dim = dim
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        self.index_path = os.path.join(base_path, "vectors.index")
        self.id_map_path = os.path.join(base_path, "id_map.json")

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

            # 🔒 DIM SAFETY CHECK (MANDATORY)
            if self.index.d != self.dim:
                raise ValueError(
                    f"Index dim {self.index.d} != expected dim {self.dim}"
                )

            if os.path.exists(self.id_map_path):
                with open(self.id_map_path, "r") as f:
                    raw_map = json.load(f)
                self.id_map = {int(k): v for k, v in raw_map.items()}
            else:
                self.id_map = {}
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.id_map: Dict[int, str] = {}

    # --------------------
    # RESET (re-ingestion)
    # --------------------
    def reset(self) -> None:
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

        if os.path.exists(self.id_map_path):
            os.remove(self.id_map_path)

        self.index = faiss.IndexFlatIP(self.dim)
        self.id_map = {}

    # --------------------
    # ADD VECTOR
    # --------------------
    def add(self, vector: np.ndarray, chunk_id: str) -> None:
        if vector.ndim != 1:
            raise ValueError("Vector must be 1D")

        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Vector dim mismatch: expected {self.dim}, got {vector.shape[0]}"
            )

        vector = vector.astype("float32").reshape(1, -1)

        # 🔒 NORMALIZATION (COSINE SAFETY)
        faiss.normalize_L2(vector)

        idx = self.index.ntotal
        self.index.add(vector)
        self.id_map[idx] = chunk_id

    # --------------------
    # SEARCH
    # --------------------
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1D")

        query_vector = query_vector.astype("float32").reshape(1, -1)

        # 🔒 NORMALIZATION (COSINE SAFETY)
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            chunk_id = self.id_map.get(int(idx))
            if chunk_id is None:
                continue

            results.append({
                "chunk_id": chunk_id,
                "score": float(score)
            })
        # print(f"✅✅ SUCCESSFULLY SEARCHED DENSE EMBEDDINGS: {len(results)}")

        return results

    # --------------------
    # PERSISTENCE
    # --------------------
    def save(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.id_map_path, "w") as f:
            json.dump(self.id_map, f)

    def size(self) -> int:
        return self.index.ntotal