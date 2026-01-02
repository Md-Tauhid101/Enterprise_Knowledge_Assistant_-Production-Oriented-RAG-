# online_pipeline.py
from graph.workflow import build_query_graph
from agents.state import QueryState
from indexes.sparse_index import BM25Index
from storage.multimodel_vector_store import MultiModalVectorStore
from retrieval.chunk_retriever import ChunkRetriever
from config import DB_CONFIG

vector_store = MultiModalVectorStore()
bm25_store = BM25Index()
# load the sparse index pickle file
bm25_store.load("sparse_store/bm25_index.pkl")

chunk_retriever = ChunkRetriever(db_config=DB_CONFIG)

def run_query(user_query: str):
    workflow = build_query_graph(vector_store=vector_store, bm25_store=bm25_store, chunk_retriever=chunk_retriever)

    initial_state: QueryState = {
        "user_query": user_query,

        # intent
        "intent": None,
        "intent_confidence": None,
        "intent_reason": None,
        "should_refuse": False,

        # rewrite
        "rewrite_candidates": None,
        "rewrite_risk": None,
        # "final_query": user_query,
        "final_query": None,

        # retrieval
        "retrieved_chunks": [],
        "final_chunks": [],

        # validation
        "validation_status": None,
        "validation_reason": None,

        # answer
        "answer_text": None,
        "answer_citations": [],
        "answer_supported": False,
        ####
        "reason": None,
    }

    # final_state = workflow.invoke(initial_state)

    # return {
    #     "answer": final_state.get("answer_text"),
    #     "citations": final_state.get("answer_citations"),
    #     # "refused": not final_state.get("answer_supported"),
    #     "refused": final_state.get("validation_status") != "pass",
    #     "reason": final_state.get("validation_reason"),
    # }

    final_state = workflow.invoke(initial_state)

    refused = (
        final_state.get("validation_status") != "pass"
        or not final_state.get("answer_supported")
    )

    reason = (
        final_state.get("validation_reason")
        or final_state.get("reason")
    )

    return {
        "answer": final_state.get("answer_text"),
        "citations": final_state.get("answer_citations"),
        "refused": refused,
        "reason": reason,
    }

if __name__ == "__main__":
    query = "Which technologies were used in the AI Agent with Web Search project?"
    response = run_query(query)

    print("\n--- RESPONSE ---")
    for k, v in response.items():
        print(f"{k}: {v}")