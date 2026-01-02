from agents.state import QueryState
from utils.llm import get_llm
from utils.json_fomater import extract_json

# def rewrite_generate_node(state: QueryState) -> QueryState:
#     if state["should_refuse"]:
#         return state

#     query = state["user_query"]
#     intent = state["intent"]
#     llm = get_llm()

#     query_candidates = []
#     risk = {
#         "precision_risk": 0.0,
#         "recall_boost": 0.0
#     }

#     if intent == "factual":
#         prompt = f"""
#         Expand the query with essential keywords only.
#         Do NOT broaden scope.

#         Query: "{query}"

#         Return JSON:
#         {{ "expanded_query": "..." }}
#         """
#         try:
#             data = extract_json(llm.invoke(prompt).content)
#             query_candidates.append(data["expanded_query"])
#             risk["recall_boost"] = 0.4
#         except Exception:
#             pass

#     elif intent == "analytical":
#         prompt = f"""
#         Generate:
#         1. A keyword-expanded query
#         2. A hypothetical answer (HyDE)

#         Rules:
#         - HyDE must only rephrase concepts implied by the query
#         - No new entities

#         Query: "{query}"

#         Return JSON:
#         {{
#           "expanded_query": "...",
#           "hyde": "..."
#         }}
#         """
#         try:
#             data = extract_json(llm.invoke(prompt).content)
#             query_candidates.extend([
#                 data["expanded_query"],
#                 data["hyde"]
#             ])
#             risk["recall_boost"] = 0.7
#             risk["precision_risk"] = 0.6
#         except Exception:
#             pass

#     elif intent == "multi_hop":
#         prompt = f"""
#         Decompose into minimal sub-questions.

#         Query: "{query}"

#         Return JSON:
#         {{ "sub_questions": ["...", "..."] }}
#         """
#         try:
#             data = extract_json(llm.invoke(prompt).content)
#             query_candidates.extend(data["sub_questions"])
#             risk["recall_boost"] = 0.8
#             risk["precision_risk"] = 0.7
#         except Exception:
#             pass
#     risk = state.get("rewrite_risk")

#     if not risk:
#         # Rewrite not available → fallback
#         return {
#             **state,
#             "final_query": state["user_query"],
#             "rewrite_decision": "fallback",
#             "rewrite_reason": "Rewrite risk unavailable"
#         }

#     return {
#         **state,
#         "rewrite_candidates": query_candidates,
#         "rewrite_risk": risk
#     }

def rewrite_generate_node(state: QueryState) -> QueryState:
    # If upstream decided this query should be refused, do nothing
    if state.get("should_refuse"):
        return state

    query = state["user_query"]
    # intent = state.get("intent", "factual")
    intent = state.get("intent")
    llm = get_llm()

    rewrite_candidates = []
    rewrite_risk = {
        "precision_risk": 0.0,
        "recall_boost": 0.0
    }

    try:
        if intent == "factual":
            prompt = f"""
            Expand the query with essential keywords only.
            Do NOT broaden scope.
            Do NOT introduce new entities.

            Query: "{query}"

            Return JSON:
            {{ "expanded_query": "..." }}
            """
            data = extract_json(llm.invoke(prompt).content)
            rewrite_candidates.append(data["expanded_query"])
            rewrite_risk["recall_boost"] = 0.4

        elif intent == "analytical":
            prompt = f"""
            Generate:
            1. A keyword-expanded query
            2. A hypothetical answer (HyDE)

            Rules:
            - HyDE must only rephrase concepts implied by the query
            - No new entities
            - No assumptions

            Query: "{query}"

            Return JSON:
            {{
              "expanded_query": "...",
              "hyde": "..."
            }}
            """
            data = extract_json(llm.invoke(prompt).content)
            rewrite_candidates.extend([
                data["expanded_query"],
                data["hyde"]
            ])
            rewrite_risk["recall_boost"] = 0.7
            rewrite_risk["precision_risk"] = 0.6

        elif intent == "multi_hop":
            prompt = f"""
            Decompose into minimal sub-questions.
            Each sub-question must be answerable independently.

            Query: "{query}"

            Return JSON:
            {{ "sub_questions": ["...", "..."] }}
            """
            data = extract_json(llm.invoke(prompt).content)
            rewrite_candidates.extend(data["sub_questions"])
            rewrite_risk["recall_boost"] = 0.8
            rewrite_risk["precision_risk"] = 0.7

    except Exception:
        # LLM failure → safe fallback
        rewrite_candidates = []
        rewrite_risk = None
    
    print(f"\n\n✅REWRITE CANDIDATES===={rewrite_candidates}\n\n")
    return {
        **state,
        "rewrite_candidates": rewrite_candidates,
        "rewrite_risk": rewrite_risk
    }

# PRECISION_RISK_THRESHOLD = 0.65

# def rewrite_guard_node(state: QueryState) -> QueryState:
#     original = state["user_query"]
#     candidates = state.get("rewrite_candidates", [])
#     risk = state.get("rewrite_risk", {})

#     precision_risk = risk.get("precision_risk", 0.0)

#     # Default: safest possible behavior
#     selected_queries = [original]
#     rewrite_allowed = False

#     if candidates and precision_risk <= PRECISION_RISK_THRESHOLD:
#         selected_queries = candidates
#         rewrite_allowed = True

#     return {
#         **state,
#         "original_query": original,
#         "selected_queries": selected_queries,
#         "rewrite_allowed": rewrite_allowed
#     }

PRECISION_RISK_THRESHOLD = 0.65

def rewrite_guard_node(state: QueryState) -> QueryState:
    original_query = state["user_query"]
    candidates = state.get("rewrite_candidates", [])
    risk = state.get("rewrite_risk", {})

    precision_risk = risk.get("precision_risk", 1.0)

    # Default: safest behavior
    final_query = original_query
    rewrite_allowed = False
    rewrite_reason = "Fallback to original query"

    if candidates and precision_risk <= PRECISION_RISK_THRESHOLD:
        final_query = candidates[0]   # SINGLE source of truth
        print(f"\n\n✅FINAL QUERY ===== {final_query}\n\n")
        rewrite_allowed = True
        rewrite_reason = "Rewrite accepted within precision risk threshold"

    return {
        **state,
        "final_query": final_query,
        "original_query": original_query,
        "rewrite_allowed": rewrite_allowed,
        "rewrite_reason": rewrite_reason
    }
