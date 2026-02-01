import argparse
import csv
import time

from src.config import settings
from src.evaluation import main as evaluate_submission


def load_questions() -> list[dict]:
    """Load questions from domande.csv. Returns list of {row_id, domanda, difficoltà}."""
    with open(settings.questions_csv_path) as f:
        reader = csv.DictReader(f)
        return [{"row_id": i + 1, **row} for i, row in enumerate(reader)]


def load_ground_truth() -> dict[int, list[int]]:
    """Load ground truth answers keyed by row_id."""
    gt = {}
    with open(settings.ground_truth_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = int(row["row_id"])
            raw = row["result"].strip()
            gt[row_id] = sorted(int(x) for x in raw.split(",") if x) if raw else []
    return gt


def build_pipeline(name: str):
    """Instantiate the chosen pipeline. All pipelines implement .query(str) -> list[int]."""
    if name == "rag":
        from src.rag import RAGPipeline

        return RAGPipeline()
    elif name == "structured":
        from src.structured_pipeline import StructuredPipeline

        return StructuredPipeline()
    else:
        raise ValueError(f"Unknown pipeline: {name}")


def run_pipeline(pipeline, questions: list[dict]) -> list[dict]:
    """Run a pipeline on all questions. Returns list of {row_id, result}."""
    ground_truth = load_ground_truth()
    results = []

    for q in questions:
        row_id = q["row_id"]
        question = q["domanda"]
        difficulty = q["difficoltà"]
        expected = ground_truth.get(row_id, [])

        print(f"\n  Q{row_id:>3} [{difficulty:<10}] {question}")

        start = time.time()
        predicted_ids = pipeline.query(question)
        elapsed = time.time() - start

        result_str = ",".join(str(d) for d in predicted_ids)
        results.append({"row_id": row_id, "result": result_str})

        print(f"       -> expected={expected}  predicted={sorted(predicted_ids)}  ({elapsed:.1f}s)")

    return results


def save_submission(results: list[dict]) -> str:
    """Save results as a submission CSV. Returns the file path."""
    submission_path = settings.output_dir / "submission.csv"
    with open(submission_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "result"])
        writer.writeheader()
        writer.writerows(results)
    return str(submission_path)


def main():
    parser = argparse.ArgumentParser(description="Run pipeline and evaluate.")
    parser.add_argument(
        "--pipeline",
        choices=["rag", "structured"],
        default="structured",
        help="Which pipeline to use (default: structured)",
    )
    args = parser.parse_args()

    print(f"=== Running {args.pipeline} pipeline ===\n")
    pipeline = build_pipeline(args.pipeline)
    questions = load_questions()
    results = run_pipeline(pipeline, questions)

    submission_path = save_submission(results)
    print(f"\nSubmission saved to: {submission_path}")

    print("\n=== Evaluating submission ===\n")
    evaluate_submission(["--submission", submission_path], standalone_mode=False)


if __name__ == "__main__":
    main()
