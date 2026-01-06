import json
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/hr/chat"

GOLDEN_FILE = Path(__file__).parent / "golden.json"


def evaluate():
    with open(GOLDEN_FILE, "r") as f:
        dataset = json.load(f)

    total = len(dataset)
    correct_intent = 0
    grounded_correct = 0

    print(f"\n🧪 Running evaluation on {total} questions...\n")

    for item in dataset:
        q = item["q"]
        expected_intent = item["expected_intent"]
        must_contain = [m.lower() for m in item["must_contain"]]

        print(f"➡️  Q: {q}")

        resp = requests.post(
            API_URL,
            json={"message": q},
            timeout=20
        )

        if resp.status_code != 200:
            print(" ❌ API error")
            continue

        data = resp.json()
        predicted = data["intent"]
        answer = (data["answer"] or "").lower()

        # intent accuracy
        intent_ok = predicted == expected_intent
        if intent_ok:
            correct_intent += 1

        # grounding check (only count if intent was correct)
        grounded_ok = False
        if intent_ok:
            grounded_ok = any(term in answer for term in must_contain)
            if grounded_ok:
                grounded_correct += 1

        print(f"   Predicted: {predicted}")
        print(f"   Expected:  {expected_intent}")
        print(f"   Intent OK? {'✅' if intent_ok else '❌'}")

        print(f"   Answer: {answer[:120]}...")
        print(f"   Grounded? {'✅' if grounded_ok else '❌'}\n")

    print("\n📊 SUMMARY")
    print(f"Intent accuracy: {correct_intent}/{total} ({round(correct_intent/total*100, 1)}%)")
    print(f"Grounded accuracy (only where intent correct): {grounded_correct}/{correct_intent} "
          f"({round((grounded_correct/max(correct_intent,1))*100, 1)}%)\n")


if __name__ == "__main__":
    evaluate()
