# multimodal_vector_store.py

from typing import Dict, List
import numpy as np

from storage.vector_store import VectorStore

# class MultiModalVectorStore:
#     """
#     Orchestrates multiple VectorStore instances
#     for different modalities (text, image, table).

#     Does NOT:
#     - create embeddings
#     - fuse results
#     - know about chunking
#     """

#     def __init__(self, base_path: str = "./vector_store"):
#         self.text_store = VectorStore(
#             dim=768,
#             base_path=f"{base_path}/text"
#         )

#         self.image_store = VectorStore(
#             dim=512,
#             base_path=f"{base_path}/image"
#         )

#         # tables use text embeddings (BGE)
#         self.table_store = VectorStore(
#             dim=768,
#             base_path=f"{base_path}/table"
#         )

#     # RESET (used during full re-ingestion)
#     def reset_all(self) -> None:
#         self.text_store.reset()
#         self.image_store.reset()
#         self.table_store.reset()

#     # ADD VECTORS
#     def add_text(self, vector: np.ndarray, chunk_id: str) -> None:
#         self.text_store.add(vector, chunk_id)

#     def add_image(self, vector: np.ndarray, chunk_id: str) -> None:
#         self.image_store.add(vector, chunk_id)

#     def add_table(self, vector: np.ndarray, chunk_id: str) -> None:
#         self.table_store.add(vector, chunk_id)

#     # SEARCH
#     def search_text(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
#         return self.text_store.search(query_vector=query_vector, top_k=top_k)
    
#     def search_image(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
#         return self.image_store.search(query_vector=query_vector, top_k=top_k)
    
#     def search_table(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
#         return self.table_store.search(query_vector=query_vector, top_k=top_k)
    
#     # PERSISTENCE
#     def save_all(self) -> None:
#         self.text_store.save()
#         self.image_store.save()
#         self.table_store.save()

class MultiModalVectorStore:
    """
    Orchestrates multiple VectorStore instances by modality.

    Responsibilities:
    - Route vectors to correct store
    - Route searches correctly
    - Manage persistence

    Does NOT:
    - Embed
    - Fuse
    - Interpret scores
    """

    def __init__(self, base_path: str = "./vector_store"):
        self.text_store = VectorStore(
            dim=768,
            base_path=f"{base_path}/text"
        )

        self.image_store = VectorStore(
            dim=512,
            base_path=f"{base_path}/image"
        )

        self.table_store = VectorStore(
            dim=768,
            base_path=f"{base_path}/table"
        )

    # --------------------
    # RESET
    # --------------------
    def reset_all(self) -> None:
        self.text_store.reset()
        self.image_store.reset()
        self.table_store.reset()

    # --------------------
    # ADD
    # --------------------
    def add_text(self, vector: np.ndarray, chunk_id: str) -> None:
        self.text_store.add(vector, chunk_id)

    def add_image(self, vector: np.ndarray, chunk_id: str) -> None:
        self.image_store.add(vector, chunk_id)

    def add_table(self, vector: np.ndarray, chunk_id: str) -> None:
        self.table_store.add(vector, chunk_id)

    # --------------------
    # SEARCH
    # --------------------
    def search_text(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        return self.text_store.search(query_vector, top_k)

    def search_image(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        return self.image_store.search(query_vector, top_k)

    def search_table(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        return self.table_store.search(query_vector, top_k)

    # --------------------
    # PERSISTENCE
    # --------------------
    def save_all(self) -> None:
        self.text_store.save()
        self.image_store.save()
        self.table_store.save()