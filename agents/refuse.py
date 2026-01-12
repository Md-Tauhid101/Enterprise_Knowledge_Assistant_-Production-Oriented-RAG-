# # agents/refusal_node.py
# from agents.state import QueryState


# def refusal_node(state: QueryState) -> dict:
#     """
#     Final decision node.
#     Produces either an answer or a refusal message.
#     """

#     # 1. Intent-level refusal
#     # if state.get("should_refuse"):
#     #     return {
#     #         "final_response": "This question can’t be answered from the available data.",
#     #         "refused": True
#     #     }

#     # 2. Validation failure
#     if state.get("validation_status") == "refuse":
#         reason = state.get("validation_reason", "")

#         if "conflict" in reason.lower():
#             msg = "The available sources provide conflicting information."
#         elif "no" in reason.lower():
#             msg = "I don’t have enough information to answer this question."
#         else:
#             msg = "This question can’t be answered from the available data."

#         return {
#             "final_response": msg,
#             "refused": True
#         }

#     # 3. Answer unsupported
#     if not state.get("answer_supported"):
#         return {
#             "final_response": "I don’t have enough information to answer this question.",
#             "refused": True
#         }

#     # 4. Success path
#     return {
#         "final_response": state["answer_text"],
#         "refused": False
#     }

# agents/refusal.py

from agents.state import QueryState

REFUSAL_MESSAGE = "The provided evidence does not contain this information."


def refusal_node(state: QueryState) -> QueryState:
    """
    Final safety gate.
    Either allows the answer or replaces it with a refusal message.
    """

    answer = (state.get("answer_text") or "").strip()
    citations = state.get("answer_citations") or []
    retrieved = state.get("retrieved_chunks") or []

    # 1️⃣ Empty answer → refuse
    if not answer:
        return {
            **state,
            "final_response": REFUSAL_MESSAGE,
            "refused": True,
            "refusal_reason": "EMPTY_ANSWER"
        }

    # 2️⃣ No retrieval happened → refuse
    if not retrieved:
        return {
            **state,
            "final_response": REFUSAL_MESSAGE,
            "refused": True,
            "refusal_reason": "NO_RETRIEVAL"
        }

    # 3️⃣ Answer given but no citations used → refuse
    if not citations:
        return {
            **state,
            "final_response": REFUSAL_MESSAGE,
            "refused": True,
            "refusal_reason": "NO_EVIDENCE"
        }

    # 4️⃣ Over-claim heuristic (simple but sane)
    if len(answer.split()) > 100 and len(citations) < 2:
        return {
            **state,
            "final_response": REFUSAL_MESSAGE,
            "refused": True,
            "refusal_reason": "OVERCLAIM"
        }
    
    # print("REFUSAL NODE DEBUG")
    # print("answer_text:", state.get("answer_text"))
    # print("citations:", state.get("answer_citations"))
    # print("retrieved:", len(state.get("retrieved_chunks") or []))


    # ✅ Allow answer
    return {
        **state,
        "final_response": answer,
        "refused": False,
        "refusal_reason": None
    }
