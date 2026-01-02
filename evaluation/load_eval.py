import json
from typing import List, Dict

def load_eval_dataset(path: str) -> List[Dict]:
    """
    Loads gold evaluation queries.
    This is read-only ground truth.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "Eval dataset must be a list"

    required_keys = {
        "id",
        "query",
        "expected_behavior",
        "expected_answer_type",
        "relevant_chunk_ids",
        "strictness"
    }

    for item in data:
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"Eval item {item.get('id')} missing keys: {missing}")

    return data

if __name__ == "__main__":
    result = load_eval_dataset("./evaluation/eval_dataset.json")
    print(result[:5])