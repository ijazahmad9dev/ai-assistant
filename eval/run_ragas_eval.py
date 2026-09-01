import json
import time
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama

from app.src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.embeddings.embedder import get_embedding_model
from eval.run_pipeline import run_after_pipeline, run_before_pipeline
from eval.hand_metrics import context_precision_at_k, faithfulness_score

GOLDEN_PATH = Path("eval/golden_dataset.json")
RAW_PATH = Path("eval/raw_runs.json")


def get_ragas_llm():
    """RAGAS's faithfulness/context_precision/context_recall metrics rely on
    the LLM returning STRUCTURED output (statement lists, per-statement JSON
    verdicts). The app's default get_llm() has no format constraint, which is
    the likely cause of those three metrics coming back NaN — RAGAS silently
    drops rows it can't parse rather than raising. format='json' pushes
    Ollama toward valid JSON; verify this against your live Ollama server, as
    local-model JSON-mode reliability varies by model."""
    return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0, format="json")


def load_golden_set():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def load_raw_runs():
    """Load whatever's already been completed, so we can resume instead of restart."""
    if RAW_PATH.exists():
        return json.loads(RAW_PATH.read_text())
    return {"after": {}, "before": {}}


def save_raw_runs(raw_runs):
    RAW_PATH.write_text(json.dumps(raw_runs, indent=2, default=str))


def run_pipeline_for_label(pipeline_fn, golden_set, label: str, raw_runs: dict):
    """Runs the pipeline for each question NOT already completed, saving after every
    single question so a crash only loses the in-flight question, not prior progress."""
    for item in golden_set:
        if item["id"] in raw_runs[label]:
            print(f"[{label}] Skipping {item['id']} (already completed)")
            continue

        print(f"[{label}] Running {item['id']}: {item['question'][:60]}...")
        result = pipeline_fn(item["question"], session_id=f"eval-{label}-{item['id']}")

        hand_precision = context_precision_at_k(
            item["question"], item["expected_answer"], result["contexts"]
        )
        # Doesn't depend on RAGAS's structured parsing, so this is always
        # populated even if RAGAS's own faithfulness metric comes back NaN.
        hand_faithfulness = faithfulness_score(result["answer"], result["contexts"])

        raw_runs[label][item["id"]] = {
            "question": item["question"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": item["expected_answer"],
            "hand_context_precision": hand_precision,
            "hand_faithfulness": hand_faithfulness,
        }
        save_raw_runs(raw_runs)   # checkpoint after EVERY question
        time.sleep(2)


from ragas.run_config import RunConfig


def run_ragas(rows):
    dataset = Dataset.from_list(rows)
    llm_wrapper = LangchainLLMWrapper(get_ragas_llm())
    embed_wrapper = LangchainEmbeddingsWrapper(get_embedding_model())

    # Your Ollama server hosts ONE model instance and processes requests
    # essentially serially. RAGAS's default RunConfig fires up to 16 concurrent
    # calls (max_workers=16) with a 180s per-call timeout — under that load,
    # requests queue up behind each other on the Ollama side and blow past
    # 180s even though each one would succeed fine on its own. max_workers=1
    # forces genuinely sequential calls (matching how Ollama processes them
    # anyway), and a much longer timeout gives a slower local model room to
    # finish faithfulness's multi-step statement extraction/verification.
    run_config = RunConfig(timeout=600, max_workers=1, max_retries=3, max_wait=5)

    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_wrapper,
        embeddings=embed_wrapper,
        run_config=run_config,
    )


def main():
    golden_set = load_golden_set()
    raw_runs = load_raw_runs()

    print("\n=== Running/resuming AFTER pipeline (CRAG graph) ===")
    run_pipeline_for_label(run_after_pipeline, golden_set, "after", raw_runs)

    print("\n=== Running/resuming BEFORE pipeline (ReAct agent) ===")
    run_pipeline_for_label(run_before_pipeline, golden_set, "before", raw_runs)

    after_rows = [raw_runs["after"][item["id"]] for item in golden_set]
    before_rows = [raw_runs["before"][item["id"]] for item in golden_set]

    print("\n=== Scoring AFTER with RAGAS ===")
    after_ragas = run_ragas(after_rows)

    print("\n=== Scoring BEFORE with RAGAS ===")
    before_ragas = run_ragas(before_rows)

    output = {
        "after": {
            "ragas": after_ragas.to_pandas().to_dict(orient="records"),
            "hand_context_precision": [
                {"id": qid, "hand_context_precision": v["hand_context_precision"]}
                for qid, v in raw_runs["after"].items()
            ],
            "hand_faithfulness": [
                {"id": qid, "hand_faithfulness": v["hand_faithfulness"]}
                for qid, v in raw_runs["after"].items()
            ],
        },
        "before": {
            "ragas": before_ragas.to_pandas().to_dict(orient="records"),
            "hand_context_precision": [
                {"id": qid, "hand_context_precision": v["hand_context_precision"]}
                for qid, v in raw_runs["before"].items()
            ],
            "hand_faithfulness": [
                {"id": qid, "hand_faithfulness": v["hand_faithfulness"]}
                for qid, v in raw_runs["before"].items()
            ],
        },
    }

    Path("eval/results.json").write_text(json.dumps(output, indent=2, default=str))
    print("\nSaved eval/results.json")
    print("\n=== AFTER (CRAG) aggregate scores ===")
    print(after_ragas)
    print("\n=== BEFORE (no CRAG) aggregate scores ===")
    print(before_ragas)


if __name__ == "__main__":
    main()