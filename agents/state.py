# state.py
from typing import TypedDict, Optional, List, Dict
import numpy as np

class QueryState(TypedDict):
    # core input
    user_query: str

    # intent understanding
    intent: Optional[str]
    intent_confidence: Optional[float]
    intent_reason: Optional[str]
    should_refuse: bool

    # rewrite generation
    rewrite_candidates: Optional[List[str]]
    rewrite_risk: Optional[Dict[str, float]]

    # rewrite guard (THIS WAS MISSING)
    original_query: Optional[str]
    selected_queries: Optional[List[str]]
    rewrite_allowed: Optional[bool]

    # Final query
    final_query: Optional[str]

    # Embedding
    query_embedding: Optional[np.ndarray]
    
    # Retrieval 
    retrieved_chunks: Optional[List[Dict]]
    retrieval_debug: Optional[Dict[str, object]]

    # Evidence validation
    final_chunks: Optional[List[Dict]]     
    validation_status: Optional[str]        
    validation_reason: Optional[str]
    
    # Answer generation
    answer_text: Optional[str]
    answer_citations: Optional[List[str]]    
    answer_supported: Optional[bool]

    # Refusal / Final output
    final_response: Optional[str]             
    refused: Optional[bool] 