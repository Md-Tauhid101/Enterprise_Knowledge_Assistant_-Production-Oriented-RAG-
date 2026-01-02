from agents.state import QueryState

MIN_FINAL_SCORE = 0.45
MAX_CONTEXT_CHARS = 6000
MAX_CHUNKS_PER_DOC = 5  #2 -> 10
MIN_TEXT_LENGTH = 20   # critical


def _is_answerable(text: str) -> bool:
    """
    Hard heuristic: prevents headers, titles, empty sections.
    This is NOT semantic understanding — just structural sanity.
    """
    print(f"✅IS ANSWERABLE: {text and len(text.strip()) >= MIN_TEXT_LENGTH}")
    return text and len(text.strip()) >= MIN_TEXT_LENGTH


def validate_node(state: QueryState, chunk_retriever) -> QueryState:

    # 0. Intent-level refusal
    # if state.get("should_refuse"):
    #     return {
    #         **state,
    #         "final_chunks": [],
    #         "validation_status": "refuse",
    #         "validation_reason": "Intent-level refusal"
    #     }

    retrieved = state.get("retrieved_chunks", [])
    if not retrieved:
        return {
            **state,
            "final_chunks": [],
            "validation_status": "refuse",
            "validation_reason": "No retrieval candidates"
        }

    # 1. Relevance threshold
    strong = [
        c for c in retrieved
        if c.get("final_score", 0.0) >= MIN_FINAL_SCORE
    ]

    if not strong:
        return {
            **state,
            "final_chunks": [],
            "validation_status": "refuse",
            "validation_reason": "No chunks above relevance threshold"
        }

    # 2. Fetch ground-truth + enforce answerability
    validated = []
    doc_counter = {}
    total_chars = 0

    for c in strong:
        chunk = chunk_retriever.get_chunk(c["chunk_id"])

        text = chunk.get("clean_text", "")
        print(f"✅RETRIEVED CHUNK: {len(text)}")
        if not _is_answerable(text):
            continue

        doc_id = chunk["document_id"]
        doc_counter.setdefault(doc_id, 0)

        if doc_counter[doc_id] >= MAX_CHUNKS_PER_DOC:
            continue

        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        validated.append({
            **c,
            "text": text,
            "document_id": doc_id
        })

        doc_counter[doc_id] += 1
        total_chars += len(text)

    # 3. Final decision
    if not validated:
        return {
            **state,
            "final_chunks": [],
            "validation_status": "refuse",
            "validation_reason": "No answerable evidence found"
        }

    return {
        **state,
        "final_chunks": validated,
        "validation_status": "pass",
        "validation_reason": None
    }
