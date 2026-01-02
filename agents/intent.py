# intent_check.py (NODE-1)
from agents.state import QueryState
from utils.llm import get_llm
from utils.json_fomater import extract_json

from langchain_core.messages import SystemMessage, HumanMessage

VALID_MODEL_INTENTS = {
    "factual",
    "analytical",
    "multi_hop",
    "unanswerable"
}

def intent_check_node(state: QueryState) -> QueryState:
    llm = get_llm()
    query = state["user_query"]

    system_prompt = """
    You are an intent classifier for a retrieval system.

    Allowed labels:
    - factual
    - analytical
    - multi_hop
    - unanswerable

    Return ONLY valid JSON.
    No markdown.
    No extra text.

    Schema:
    {
      "intent": string,
      "confidence": number,
      "reason": string
    }
    """

    user_prompt = f'Query: "{query}"'

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])


    try:
        parsed = extract_json(response.content)
        print(f"✅INTENT CLASSIFICATION===={parsed}")
    except Exception:
        return {
            **state,
            "intent": "unknown",
            "intent_confidence": 0.0,
            "intent_reason": "Invalid LLM output",
            # "should_refuse": False,
        }
    #################################
    intent = parsed.get("intent", "").strip().lower()
    if intent not in VALID_MODEL_INTENTS:
        intent = "unknown"

    confidence = float(parsed.get("confidence", 0.0))
    ######################################
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "intent_reason": parsed.get("reason", ""),
        # "should_refuse": (intent == "unanswerable" and confidence >= 0.75)  # early flag to refuse before refusal logic
    }
