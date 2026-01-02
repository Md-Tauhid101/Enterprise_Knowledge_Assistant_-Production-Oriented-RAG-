import json
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm import get_llm
import time

RATE_LIMIT_SECONDS = 15

llm = get_llm()

SYSTEM_PROMPT = """
        You are an evidence-bound assistant.
        Rules:
        - Answer ONLY using the provided evidence.
        - Do NOT add external knowledge.
        - Do NOT infer or speculate.
        - If the evidence does not answer the question, say exactly:
        "The provided evidence does not contain this information."
        - Keep the answer as short as possible.
    """

def generate_eval_answer(query: str, retrieved_chunks):
    """
    Offline answer generation for evaluation.
    No validation, no refusal routing, no rewrite.
    """

    if not retrieved_chunks:
        return "The provided evidence not contain this information."
    
    context = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}"
        for c in retrieved_chunks
        if c.get("text")
    )

    user_prompt = f"""
        Question:
        {query}

        Evidence:
        {context}

        Ansawer:
    """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    return response.content.strip()

def run_eval_answer_generation(eval_retrieval_results, output_path):
    """
    Generates answers for all eval queries and stores them to JSON.
    """
    outputs = []

    for item in eval_retrieval_results:
        behavior = item["expected_behavior"]
        # --- CASE 1: ANSWERABLE → LLM call ---
        if behavior == "answer":
            answer = generate_eval_answer(
                query=item["query"],
                retrieved_chunks=item["retrieved_chunks"]
            )
            time.sleep(RATE_LIMIT_SECONDS)

        # --- CASE 2: REFUSE → deterministic ---
        elif behavior == "refuse":
            answer = "The provided evidence does not contain this information."

        # --- CASE 3: AMBIGUOUS → skip for now ---
        elif behavior == "ambiguous":
            answer = "__SKIPPED_AMBIGUOUS__"

        else:
            raise ValueError(f"Unknown expected_behavior: {behavior}")

        outputs.append({
            "eval_id": item["eval_id"],
            "query": item["query"],
            "expected_behavior": behavior,
            "expected_answer_type": item["expected_answer_type"],
            "strictness": item["strictness"],
            "answer_text": answer,
            "retrieved_chunks": item["retrieved_chunks"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    return outputs

if __name__ == "__main__":
    def load_retrieval_results(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def main():
        from evaluation.load_eval import load_eval_dataset

        print("🔍 Loading evaluation dataset...")
        eval_data = load_eval_dataset("evaluation/eval_dataset.json")
        print(f"✅ Loaded {len(eval_data)} eval queries\n")


        print("📦 Loading retrieval results...")
        
        retrieval_results = load_retrieval_results(
            "evaluation/eval_retrieval_results.json"
        )

        print("🧠 Generating eval answers (offline)...")
        outputs = run_eval_answer_generation(
            eval_retrieval_results=retrieval_results,
            output_path="evaluation/eval_outputs.json"
        )

        print(f"✅ Generated {len(outputs)} eval answers")
        print("📁 Saved to evaluation/eval_outputs.json")
    
    main()