# agents/refusal_node.py
from agents.state import QueryState


def refusal_node(state: QueryState) -> dict:
    """
    Final decision node.
    Produces either an answer or a refusal message.
    """

    # 1. Intent-level refusal
    # if state.get("should_refuse"):
    #     return {
    #         "final_response": "This question can’t be answered from the available data.",
    #         "refused": True
    #     }

    # 2. Validation failure
    if state.get("validation_status") == "refuse":
        reason = state.get("validation_reason", "")

        if "conflict" in reason.lower():
            msg = "The available sources provide conflicting information."
        elif "no" in reason.lower():
            msg = "I don’t have enough information to answer this question."
        else:
            msg = "This question can’t be answered from the available data."

        return {
            "final_response": msg,
            "refused": True
        }

    # 3. Answer unsupported
    if not state.get("answer_supported"):
        return {
            "final_response": "I don’t have enough information to answer this question.",
            "refused": True
        }

    # 4. Success path
    return {
        "final_response": state["answer_text"],
        "refused": False
    }
