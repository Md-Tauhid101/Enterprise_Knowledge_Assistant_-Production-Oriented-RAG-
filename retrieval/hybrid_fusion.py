# retrieval/hybrid_fusion.py
from typing import List, Dict

INTENT_WEIGHTS = {
    "factual":       (0.4, 0.6),
    "explanatory":   (0.6, 0.4),
    "ambiguous":     (0.5, 0.5)
}

def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}

    min_s = min(scores.values())
    max_s = max(scores.values())

    if min_s == max_s:
        return {k: 1.0 for k in scores}

    return {
        k: (v - min_s) / (max_s - min_s)
        for k, v in scores.items()
    }


# def hybrid_fusion(dense_results: List[Dict], sparse_results: List[Dict], intent: str, top_k: int = 10 ) -> List[Dict]:
#     print("DENSE TOP 5")
#     for r in dense_results[:5]:
#         print(r["chunk_id"], r["dense_score"])

#     print("SPARSE TOP 5")
#     for r in sparse_results[:5]:
#         print(r["chunk_id"], r["sparse_score"])

#     dense_scores = {r["chunk_id"]: r["dense_score"] for r in dense_results}
#     sparse_scores = {r["chunk_id"]: r["sparse_score"] for r in sparse_results}

#     dense_norm = min_max_normalize(dense_scores)
#     sparse_norm = min_max_normalize(sparse_scores)

#     w_dense, w_sparse = INTENT_WEIGHTS.get(
#         intent, INTENT_WEIGHTS["ambiguous"]
#     )

#     all_chunk_ids = set(dense_norm) | set(sparse_norm)

#     fused = []
#     for cid in all_chunk_ids:
#         fused.append({
#             "chunk_id": cid,
#             "dense_score": dense_norm.get(cid, 0.0),
#             "sparse_score": sparse_norm.get(cid, 0.0),
#             "final_score": (
#                 w_dense * dense_norm.get(cid, 0.0) + w_sparse * sparse_norm.get(cid, 0.0)
#             )
#         })

#     fused.sort(key=lambda x: x["final_score"], reverse=True)
#     return fused[:top_k]


def hybrid_fusion(dense_results, sparse_results, intent, top_k=10):

    dense_scores = {r["chunk_id"]: r["dense_score"] for r in dense_results}
    sparse_scores = {r["chunk_id"]: r["sparse_score"] for r in sparse_results}

    dense_norm = min_max_normalize(dense_scores)
    sparse_norm = min_max_normalize(sparse_scores)

    if intent == "factual":
        w_dense, w_sparse = 0.8, 0.2
        DENSE_GATE = 0.35
    else:
        w_dense, w_sparse = 0.6, 0.4
        DENSE_GATE = 0.0

    fused = []
    for cid in dense_norm:
        if dense_norm[cid] < DENSE_GATE:
            continue

        fused.append({
            "chunk_id": cid,
            "final_score": (
                w_dense * dense_norm[cid] +
                w_sparse * sparse_norm.get(cid, 0.0)
            )
        })

    fused.sort(key=lambda x: x["final_score"], reverse=True)
    return fused[:top_k]

