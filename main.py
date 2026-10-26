import argparse
import json
import sys
from pathlib import Path

try:
    from .models import GroundTruthItem, PredictionItem
    from .evaluator import evaluate_batch
except ImportError:
    from models import GroundTruthItem, PredictionItem
    from evaluator import evaluate_batch


def run_evaluation(data_path: Path, iou_threshold: float = 0.5):
    if not data_path.exists():
        print(f"Error: Dataset file not found at {data_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truths = [GroundTruthItem(**gt) for gt in data.get("ground_truths", [])]
    predictions = [PredictionItem(**pred) for pred in data.get("predictions", [])]

    summary = evaluate_batch(ground_truths, predictions, iou_threshold=iou_threshold)

    print("=" * 70)
    print("VLM Spatial Grounding Evaluation Report")
    print("=" * 70)
    print(f"Dataset:        {data_path.name}")
    print(f"Total Samples:  {summary.total_samples}")
    print(f"IoU Threshold:  {summary.iou_threshold}")
    print(f"Mean IoU:       {summary.mean_iou:.4f}")
    print(f"Accuracy @ IoU: {summary.accuracy_at_threshold * 100:.1f}% ({summary.correct_samples}/{summary.total_samples})")
    print("-" * 70)
    print(f"{'Item ID':<12} | {'Image ID':<18} | {'IoU':<8} | {'Label Match':<11} | {'Status'}")
    print("-" * 70)
    for r in summary.results:
        label_str = "YES" if r.label_match else "NO"
        print(f"{r.item_id:<12} | {r.image_id:<18} | {r.iou:<8.4f} | {label_str:<11} | {r.status}")
    print("=" * 70)

    return summary


def main():
    default_dataset = Path(__file__).parent / "sample_dataset.json"
    parser = argparse.ArgumentParser(description="Evaluate VLM spatial grounding predictions against ground truth labels.")
    parser.add_argument("--data", type=Path, default=default_dataset, help="Path to evaluation JSON dataset.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for accuracy calculation (default 0.5).")
    args = parser.parse_args()

    run_evaluation(args.data, args.iou_threshold)


if __name__ == "__main__":
    main()
