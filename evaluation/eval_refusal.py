import json

REFUSAL_PHRASES = [
    "does not contain this information",
    "not mentioned",
    "not provided",
    "no information",
    "cannot be determined"
]


def is_refusal(answer_text: str) -> bool:
    if not answer_text:
        return True

    text = answer_text.lower()
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def refusal_correctness(eval_outputs, eval_dataset):
    eval_map = {item["id"]: item for item in eval_dataset}

    total = 0
    correct = 0
    errors = []

    for item in eval_outputs:
        eval_id = item["eval_id"]
        answer = item["answer_text"]

        gold = eval_map.get(eval_id)
        if not gold:
            continue

        expected = gold["expected_behavior"]

        # Skip ambiguous
        if expected == "ambiguous":
            continue

        total += 1

        refused = is_refusal(answer)

        is_correct = (
            (expected == "refuse" and refused) or
            (expected == "answer" and not refused)
        )

        if is_correct:
            correct += 1
        else:
            errors.append({
                "eval_id": eval_id,
                "expected": expected,
                "answer": answer
            })

    return {
        "refusal_accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "errors": errors
    }


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("📦 Loading eval dataset...")
    eval_data = load("evaluation/eval_dataset.json")

    print("📦 Loading eval outputs...")
    eval_outputs = load("evaluation/eval_outputs.json")

    print("🛑 Running refusal correctness evaluation...\n")

    results = refusal_correctness(eval_outputs, eval_data)

    print("📊 Refusal Correctness Summary")
    print("------------------------------")
    print(f"Total evaluated : {results['total']}")
    print(f"Correct         : {results['correct']}")
    print(f"Accuracy        : {results['refusal_accuracy']:.3f}")

    if results["errors"]:
        print("\n❌ Refusal errors:\n")
        for e in results["errors"]:
            print(f"[{e['eval_id']}] expected={e['expected']}")
            print("Answer:", e["answer"])
            print("-" * 60)
    else:
        print("\n✅ No refusal errors detected.")

    print("\n🎯 Refusal correctness evaluation completed.")


if __name__ == "__main__":
    main()