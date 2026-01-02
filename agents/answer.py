# agents/answer_node.py
from agents.state import QueryState
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import get_llm

llm = get_llm()


def answer_generation_node(state: QueryState) -> QueryState:

    # Hard stop — validator decides authority
    if state.get("validation_status") != "pass":
        return {
            **state,
            "answer_text": None,
            "answer_citations": [],
            "answer_supported": False,
            "refused": True,
            "reason": state.get("validation_reason", "Validation failed")
        }

    chunks = state.get("final_chunks", [])
    if not chunks:
        return {
            **state,
            "answer_text": None,
            "answer_citations": [],
            "answer_supported": False,
            "refused": True,
            "reason": "No validated evidence"
        }

    query = state["final_query"]

    context = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}"
        for c in chunks
    )

    # print(f"✅CONTEXT: {context}...")

    system_prompt = (
        "You are an evidence-bound assistant.\n\n"
        "Rules:\n"
        "- Answer ONLY using the provided evidence.\n"
        "- Do NOT add external knowledge.\n"
        "- If the evidence does not fully answer the question, say so explicitly.\n"
        "- Do NOT speculate or infer beyond the text."
    )

    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Evidence:\n{context}\n\n"
        "Answer:"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    answer_text = response.content.strip()


    # Explicit refusal detection
    refusal_signals = [
        "insufficient",
        "does not answer",
        "not provided",
        "cannot be determined",
        "not mentioned"
    ]

    if any(sig in answer_text.lower() for sig in refusal_signals):
        return {
            **state,
            "answer_text": None,
            "answer_citations": [],
            "answer_supported": False,
            "refused": True,
            "reason": "Evidence insufficient to answer question"
        }

    citations = [c["chunk_id"] for c in chunks]

    return {
        **state,
        "answer_text": answer_text,
        "answer_citations": citations,
        "answer_supported": True,
        "refused": False,
        "reason": None
    }
