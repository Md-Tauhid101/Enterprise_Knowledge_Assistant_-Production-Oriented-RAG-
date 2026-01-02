from agents.state import QueryState
from indexes.dense_embeddings import embed_texts

def query_embedding_node(state: QueryState) -> QueryState:
    query = state["final_query"]

    if not query:
        return {
            **state,
            "query_embedding": None
        }

    embedding = embed_texts([query])[0]
    print("✅✅SUCCESSFULLY GENERATED EMBEDDINGS")


    return {
        **state,
        "query_embedding": embedding
    }
