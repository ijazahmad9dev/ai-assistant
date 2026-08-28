import json
from pathlib import Path

data = json.loads(Path("eval/results.json").read_text())


def avg(records, key):
    vals = [r[key] for r in records if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def avg_hand(records, key):
    vals = [r[key] for r in records]
    return sum(vals) / len(vals) if vals else 0.0


metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

print(f"{'Metric':<22}{'Before':>10}{'After':>10}{'Delta':>10}")
print("-" * 52)
for m in metrics:
    before_val = avg(data["before"]["ragas"], m)
    after_val = avg(data["after"]["ragas"], m)
    delta = after_val - before_val
    print(f"{m:<22}{before_val:>10.3f}{after_val:>10.3f}{delta:>+10.3f}")

before_hand_prec = avg_hand(data["before"]["hand_context_precision"], "hand_context_precision")
after_hand_prec = avg_hand(data["after"]["hand_context_precision"], "hand_context_precision")
print(f"{'hand_context_prec.':<22}{before_hand_prec:>10.3f}{after_hand_prec:>10.3f}{after_hand_prec - before_hand_prec:>+10.3f}")

before_hand_faith = avg_hand(data["before"]["hand_faithfulness"], "hand_faithfulness")
after_hand_faith = avg_hand(data["after"]["hand_faithfulness"], "hand_faithfulness")
print(f"{'hand_faithfulness':<22}{before_hand_faith:>10.3f}{after_hand_faith:>10.3f}{after_hand_faith - before_hand_faith:>+10.3f}")