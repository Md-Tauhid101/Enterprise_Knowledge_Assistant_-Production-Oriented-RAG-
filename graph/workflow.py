# graph/workflow.py
from langgraph.graph import StateGraph, END
from agents.state import QueryState

from agents.intent import intent_check_node
from agents.rewrite import rewrite_generate_node, rewrite_guard_node
from agents.retrieve import retrieve_node
from agents.validate import validate_node
from agents.answer import answer_generation_node
from agents.query_embedding import query_embedding_node
from agents.refuse import refusal_node
from functools import partial


def build_query_graph(vector_store, bm25_store, chunk_retriever):
    graph = StateGraph(QueryState)

    # retrieve
    retrieve = partial(
        retrieve_node,
        vector_store=vector_store,
        bm25_store=bm25_store
    )

    validate = partial(
        validate_node,
        chunk_retriever= chunk_retriever
    )

    # Nodes
    graph.add_node("intent_check", intent_check_node)
    graph.add_node("rewrite_generate", rewrite_generate_node)
    graph.add_node("rewrite_guard", rewrite_guard_node)
    graph.add_node("query_embedding", query_embedding_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("validate", validate)
    graph.add_node("answer", answer_generation_node)
    # graph.add_node("refuse", refusal_node)

    # Edges (linear part)
    graph.set_entry_point("intent_check")
    graph.add_edge("intent_check", "rewrite_generate")
    graph.add_edge("rewrite_generate", "rewrite_guard")
    graph.add_edge("rewrite_guard", "query_embedding")
    graph.add_edge("query_embedding", "retrieve")
    graph.add_edge("retrieve", "validate")

    graph.add_conditional_edges(
        "validate",
        lambda state: "answer" if state.get("validation_status") == "pass" else END,
        {
            "answer": "answer",
            END: END
        }
    )


    # # Conditional branching
    # def validation_router(state: QueryState):
    #     if state.get("validation_status") == "pass":
    #         return "answer"
    #     return "refuse"

    # graph.add_conditional_edges(
    #     "validate",
    #     validation_router,
    #     {
    #         "answer": "answer",
    #         "refuse": "refuse",
    #     }
    # )

    graph.add_edge("answer", END)
    # graph.add_edge("refuse", END)

    return graph.compile()

if __name__ == "__main__":
    
    from indexes.sparse_index import BM25Index
    from storage.multimodel_vector_store import MultiModalVectorStore
    from retrieval.chunk_retriever import ChunkRetriever
    from config import DB_CONFIG

    vector_store = MultiModalVectorStore()
    bm25_store = BM25Index()
    # load the sparse index pickle file
    bm25_store.load("sparse_store/bm25_index.pkl")

    chunk_retriever = ChunkRetriever(db_config=DB_CONFIG)
    graph = build_query_graph(vector_store=vector_store, bm25_store=bm25_store, chunk_retriever=chunk_retriever)
    print(graph.get_graph().draw_mermaid())