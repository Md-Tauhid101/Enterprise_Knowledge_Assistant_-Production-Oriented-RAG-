import random
import time

import json
from evaluation.load_eval import load_eval_dataset

from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import get_llm

llm = get_llm()

def token_overlap_ratio(answer, context):
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def deterministic_faithfulness(
    answer_text,
    retrieved_chunks,
    expected_behavior,
    expected_answer_type,
    strictness
):
    context_text = " ".join(c["text"] for c in retrieved_chunks)

    # Empty answer handling
    if not answer_text:
        return expected_behavior != "answer"

    # check extractive answer -> no creativity allowed in answer
    if expected_answer_type == "extractive":
        return answer_text.strip() in context_text

    # Token overlap check(answer introduce too many tokens not present in evidence.)
    overlap = token_overlap_ratio(answer_text, context_text)

    thresholds = {
        "high": 0.8,
        "medium": 0.6,
        "low": 0.4
    }

    if overlap < thresholds[strictness]:
        return False

    # overexplaination beyond evidence
    if len(answer_text) > 2 * len(context_text):
        return False

    return True


def evaluate_faithfulness(eval_results: list[dict]):
    total = 0
    faithful = 0
    faithfulness_passes = []
    faithfulness_failures = []

    for item in eval_results:
        if item["expected_behavior"] != "answer":
            continue

        total += 1

        is_faithful = deterministic_faithfulness(
            answer_text=item["answer_text"],
            retrieved_chunks=item["retrieved_chunks"],
            expected_behavior=item["expected_behavior"],
            expected_answer_type=item["expected_answer_type"],
            strictness=item["strictness"]
        )

        if is_faithful:
            faithfulness_passes.append({
                "eval_id": item["eval_id"],
                "query": item["query"],
                "answer": item["answer_text"]
            })
            faithful += 1
        else:
            faithfulness_failures.append({
                "eval_id": item["eval_id"],
                "query": item["query"],
                "answer": item["answer_text"]
            })

    return {
        "faithfulness_rate": faithful / total if total else 0.0,
        "total": total,
        "faithful": faithful,
        "faithfulness_passes": faithfulness_passes,
        "faithfulness_failures": faithfulness_failures
    }
    
SYSTEM_PROMPT = "You are a strict evidence auditor."
def llm_faithfulness_check(query, answer, retrieved_chunks):
    context = "\n\n".join(
        c["text"] for c in retrieved_chunks if c.get("text")
    )
    
    user_promt = f"""
    Question:
    {query}
    
    Ansawer:
    {answer}
    
    Evidence:
    {context}

    Is the answer fully supported by the evidence?
    
    Reply with only one word:
    Yes or No
    """
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_promt)
    ]
    
    response = llm.invoke(messages)
    return response.content.strip().upper()

RATE_LIMIT_SECONDS = 15
def run_llm_faithfulness(
    eval_outputs,
    deterministic_failures,
    sample_rate=0.2,
    max_calls=8
    ):
    
    """
    Runs LLM faithfulness on:
    - all deterministic failures
    - a random sample of passes
    """
    
    candidates = []

    # 1. Always audit deterministic failures
    for item in eval_outputs:
        if item["eval_id"] in deterministic_failures:
            candidates.append(item)

    # 2. Sample from passes
    passes = [
        item for item in eval_outputs
        if item["eval_id"] not in deterministic_failures
        and item["expected_behavior"] == "answer"
    ]

    sample_size = int(len(passes) * sample_rate)
    candidates.extend(random.sample(passes, min(sample_size, len(passes))))

    # 3. Hard cap (quota safety)
    candidates = candidates[:max_calls]

    results = []

    for item in candidates:
        verdict = llm_faithfulness_check(
            query=item["query"],
            answer=item["answer_text"],
            retrieved_chunks=item["retrieved_chunks"]
        )

        results.append({
            "eval_id": item["eval_id"],
            "llm_faithful": verdict
        })

        time.sleep(RATE_LIMIT_SECONDS)

    return results
    



def load_eval_outputs(path: str):
    """
    Loads evaluation outputs AFTER answer generation.
    Each item must contain:
    - eval_id
    - answer_text
    - retrieved_chunks
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("🔍 Loading eval dataset...")
    eval_data = load_eval_dataset("evaluation/eval_dataset.json")

    print("📦 Loading eval outputs (with answers)...")
    eval_outputs = load_eval_outputs("evaluation/eval_outputs.json")

    print("\n🧪 Running deterministic faithfulness check...\n")
        
    result = evaluate_faithfulness(eval_results=eval_outputs)
    with open("./evaluation/faithful_passes.json", "w", encoding="utf-8") as f:
        json.dump(result["faithfulness_passes"], f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"Saved pass cases")
    print("=" * 60)
    with open("./evaluation/faithful_failure.json", "w", encoding="utf-8") as f:
        json.dump(result["faithfulness_failures"], f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 60)
    print(f"Saved failure cases")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("📊 FAITHFULNESS EVALUATION REPORT")
    print("=" * 60)

    print(f"Total samples        : {result['total']}")
    print(f"Faithful responses   : {result['faithful']}")
    print(f"Faithfulness rate    : {result['faithfulness_rate']:.2%}")

    print("\n✅ PASSED CASES")
    print("-" * 60)
    if not result["faithfulness_passes"]:
        print("None")
    else:
        for i, p in enumerate(result["faithfulness_passes"], 1):
            print(f"{i}. Eval ID: {p.get('eval_id')}")
            print(f"   Query : {p.get('query')}")
            print()

    print("\n❌ FAILED CASES")
    print("-" * 60)
    if not result["faithfulness_failures"]:
        print("None")
    else:
        for i, f in enumerate(result["faithfulness_failures"], 1):
            print(f"{i}. Eval ID: {f.get('eval_id')}")
            print(f"   Query : {f.get('query')}")
            print(f"   Reason: {f.get('reason', 'unknown')}")
            print()

    print("=" * 60)

    print("\n🎯 Deterministic faithfulness check completed.")
    
def main1():
        # ---- CONFIG ----
    RATE_LIMIT_SECONDS = 15          # Gemini free tier safe
    MAX_CALLS = 8                    # hard cap (important)
    TARGET_EVAL_IDS = {
        "EVAL_A_01",
        "EVAL_A_06",
        "EVAL_A_03",
        "EVAL_A_07",
        "EVAL_A_08",
        "EVAL_A_10",
    }

    print("📦 Loading eval outputs...")
    eval_outputs = load_eval_outputs("evaluation/eval_outputs.json")

    # Filter only target cases
    candidates = [
        item for item in eval_outputs
        if item["eval_id"] in TARGET_EVAL_IDS
    ]

    candidates = candidates[:MAX_CALLS]

    print(f"🧪 Running LLM faithfulness on {len(candidates)} cases\n")

    results = []

    for item in candidates:
        print(f"▶ Checking {item['eval_id']}")
        print("Query:", item["query"])
        print("Answer:", item["answer_text"])

        verdict = llm_faithfulness_check(
            query=item["query"],
            answer=item["answer_text"],
            retrieved_chunks=item["retrieved_chunks"]
        )

        results.append({
            "eval_id": item["eval_id"],
            "llm_faithful": verdict
        })

        print("LLM verdict:", verdict)
        print("-" * 60)

        time.sleep(RATE_LIMIT_SECONDS)

    # ---- Summary ----
    yes = sum(1 for r in results if r["llm_faithful"] == "YES")
    no = sum(1 for r in results if r["llm_faithful"] == "NO")

    print("\n📊 LLM Faithfulness Summary")
    print("--------------------------")
    print(f"YES: {yes}")
    print(f"NO : {no}")

    print("\n🎯 LLM faithfulness audit completed.")
    
if __name__ == "__main__":
    main1()