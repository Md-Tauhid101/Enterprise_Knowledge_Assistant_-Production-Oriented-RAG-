

def recall_at_k(retrieved_chunk_ids, gold_chunk_ids, k):
    if not gold_chunk_ids:
        return None  # not applicable

    retrieved_top_k = set(retrieved_chunk_ids[:k])
    gold = set(gold_chunk_ids)

    return int(len(retrieved_top_k & gold) > 0)


def compute_recall_at_k(eval_results, k=5):
    total = 0
    hits = 0
    per_query = []

    for item in eval_results:
        if item["expected_behavior"] != "answer":
            continue

        score = recall_at_k(
            retrieved_chunk_ids=item["retrieved_chunk_ids"],
            gold_chunk_ids=item["gold_chunk_ids"],
            k=k
        )

        if score is None:
            continue

        total += 1
        hits += score

        per_query.append({
            "eval_id": item["eval_id"],
            "query": item["query"],
            "recall_at_k": score
        })

    recall = hits / total if total > 0 else 0.0

    return {
        "recall_at_k": recall,
        "total_answerable": total,
        "hits": hits,
        "per_query": per_query
    }

if __name__ == "__main__":
    from evaluation.load_eval import load_eval_dataset
    from evaluation.run_retrieval import (
        run_retrieval_eval,
        run_retrieval_on_eval_set
    )

    TOP_K = 5

    print("🔍 Loading evaluation dataset...")
    eval_data = load_eval_dataset("evaluation/eval_dataset.json")
    print(f"✅ Loaded {len(eval_data)} eval queries\n")
    eval_results = run_retrieval_on_eval_set(eval_data=eval_data, top_k=TOP_K)
    recall_metrics = compute_recall_at_k(eval_results=eval_results, k = TOP_K)
    print("Recall@5:", recall_metrics["recall_at_k"])
    print("Hits:", recall_metrics["hits"])
    print("Total answerable:", recall_metrics["total_answerable"])