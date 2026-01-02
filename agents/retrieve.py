# agents/retrieve_node.py
from agents.state import QueryState
from retrieval.retrieval_signal import dense_retrieve_text, sparse_retrieve
from retrieval.hybrid_fusion import hybrid_fusion


FINAL_TOP_K = 8


def retrieve_node(state: QueryState, vector_store, bm25_store) -> QueryState:

    # MUST use guarded query
    query = state["final_query"]
    query_embedding = state.get("query_embedding")
    # intent = state.get("intent", "ambiguous")
    intent = state.get("intent", "unknown")

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
        top_k=FINAL_TOP_K
    )

    return {
        **state,
        "retrieved_chunks": fused,
        "retrieval_debug": {
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fusion": "intent_weighted_minmax"
        }
    }
