import json
from pathlib import Path

data = json.loads(Path("eval/results.json").read_text())


def avg(records, key):
    vals = [r[key] for r in records if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def avg_hand(records):
    vals = [r["hand_context_precision"] for r in records]
    return sum(vals) / len(vals) if vals else 0.0


metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

print(f"{'Metric':<22}{'Before':>10}{'After':>10}{'Delta':>10}")
print("-" * 52)
for m in metrics:
    before_val = avg(data["before"]["ragas"], m)
    after_val = avg(data["after"]["ragas"], m)
    delta = after_val - before_val
    print(f"{m:<22}{before_val:>10.3f}{after_val:>10.3f}{delta:>+10.3f}")

before_hand = avg_hand(data["before"]["hand_context_precision"])
after_hand = avg_hand(data["after"]["hand_context_precision"])
print(f"{'hand_context_prec.':<22}{before_hand:>10.3f}{after_hand:>10.3f}{after_hand - before_hand:>+10.3f}")