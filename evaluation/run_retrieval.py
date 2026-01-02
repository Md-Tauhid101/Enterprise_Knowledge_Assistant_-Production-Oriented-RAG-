from typing import Dict, List
from evaluation.eval_retriever import eval_retriver
from storage.multimodel_vector_store import MultiModalVectorStore
from indexes.sparse_index import BM25Index

vector_store = MultiModalVectorStore()
bm25_store = BM25Index()
# load the sparse index pickle file
bm25_store.load("sparse_store/bm25_index.pkl")

def run_retrieval_eval(query: str, top_k ) -> Dict:
    """
    Runs retrieval ONLY.
    No LLM.
    No answer generation.
    """

    results = eval_retriver(
            query=query,
            vector_store=vector_store,
            bm25_store=bm25_store,
            intent="unknown",
            top_k=top_k
        )

    retrieved_chunk_ids = []
    retrieved_chunks = []

    for r in results:
        retrieved_chunk_ids.append(r["chunk_id"])
        retrieved_chunks.append({
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "score": r.get("score")
        })

    return {
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_chunks": retrieved_chunks,
        "top_k": top_k
    }

def run_retrieval_on_eval_set(eval_data, top_k):
    outputs = []

    for item in eval_data:
        retrieval = run_retrieval_eval(
            query=item["query"],
            top_k=top_k
        )

        outputs.append({
            "eval_id": item["id"],
            "query": item["query"],
            "expected_behavior": item["expected_behavior"],
            "expected_answer_type": item["expected_answer_type"],
            "strictness": item["strictness"],
            "gold_chunk_ids": item["relevant_chunk_ids"],
            "retrieved_chunk_ids": retrieval["retrieved_chunk_ids"],
            "retrieved_chunks": retrieval["retrieved_chunks"]
        })

    return outputs


if __name__ == "__main__":
    import json
    from evaluation.load_eval import load_eval_dataset
    from evaluation.run_retrieval import (
        run_retrieval_eval,
        run_retrieval_on_eval_set
    )

    def save_json(data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def format_retrieval_results(all_results):
        """
        Strip score and keep only chunk_id + text.
        """
        formatted = []

        for r in all_results:
            cleaned_chunks = [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"]
                }
                for c in r["retrieved_chunks"]
            ]

            formatted.append({
                "eval_id": r["eval_id"],
                "query": r["query"],
                "expected_behavior": r["expected_behavior"],
                "expected_answer_type": r["expected_answer_type"],
                "strictness": r["strictness"],
                "retrieved_chunks": cleaned_chunks
            })

        return formatted



    print("🔍 Loading evaluation dataset...")
    eval_data = load_eval_dataset("evaluation/eval_dataset.json")
    print(f"✅ Loaded {len(eval_data)} eval queries\n")



    # ---- RUN FULL EVAL SET (RETRIEVAL ONLY) ----
    print("📦 Running retrieval on full eval set...\n")
    all_results = run_retrieval_on_eval_set(
        eval_data=eval_data,
        top_k=4
    )
    formatted_results = format_retrieval_results(all_results)

    save_json(
        formatted_results,
        "./evaluation/eval_retrieval_results.json"
    )

    print("📁 Saved retrieval results to evaluation/eval_retrieval_results.json")


    print(f"✅ Retrieval completed for {len(all_results)} queries\n")

    # ---- QUICK SANITY PRINT (FIRST 3) ----
    # for r in all_results[:1]:
    #     print(f"[{r['eval_id']}] {r['query']}")
    #     print("  Gold:", r["gold_chunk_ids"])
    #     print("  Retrieved:", r["retrieved_chunk_ids"])
    #     print("  Retrieved chunks:", r["retrieved_chunks"])
    #     print()
    #     # save_json(r["retrieved_chunks"], "./evaluation/eval_retrieval_results.json")

    # print("🎯 Retrieval evaluation setup is working correctly.")
    
