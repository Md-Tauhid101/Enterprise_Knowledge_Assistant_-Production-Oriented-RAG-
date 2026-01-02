# eval_retriever.py
from retrieval.retrieval_signal import dense_retrieve_text, sparse_retrieve
from retrieval.hybrid_fusion import hybrid_fusion
from indexes.dense_embeddings import embed_texts
from retrieval.chunk_retriever import ChunkRetriever
from config import DB_CONFIG

chunk_retriever = ChunkRetriever(DB_CONFIG)

def eval_retriver(query, vector_store, bm25_store, intent, top_k):
    query = query
    query_embedding = embed_texts([query])[0]

    dense_results = dense_retrieve_text( # output = List[Dict]
        query_embedding=query_embedding,
        vector_store=vector_store,
        top_k=15
    )

    sparse_results = sparse_retrieve(
        query=query,
        bm25_index=bm25_store,
        top_k=10
    )

    fused = hybrid_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        intent=intent,
        top_k=top_k
    )

    results = []
    for f in fused:
        chunk_text = chunk_retriever.get_chunk(f["chunk_id"])["clean_text"]
        results.append({
            "chunk_id": f["chunk_id"],
            "text": chunk_text,
            "score": f["final_score"]
        })
    return results

if __name__ == "__main__":
    from storage.multimodel_vector_store import MultiModalVectorStore
    from indexes.sparse_index import BM25Index

    vector_store = MultiModalVectorStore()
    bm25_store = BM25Index()
    # load the sparse index pickle file
    bm25_store.load("sparse_store/bm25_index.pkl")

    eval_retriver(query="Which technologies were used in the AI Agent with Web Search project?",
                  vector_store=vector_store,
                  bm25_store=bm25_store,
                  intent="unknown",
                  top_k=5)