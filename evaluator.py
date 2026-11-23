from typing import Sequence

try:
    from .models import BoundingBox, GroundTruthItem, PredictionItem, EvalResult, EvaluationSummary
except ImportError:
    from models import BoundingBox, GroundTruthItem, PredictionItem, EvalResult, EvaluationSummary


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calculates Intersection over Union (IoU) for two 2D bounding boxes."""
    inter_ymin = max(box_a.ymin, box_b.ymin)
    inter_xmin = max(box_a.xmin, box_b.xmin)
    inter_ymax = min(box_a.ymax, box_b.ymax)
    inter_xmax = min(box_a.xmax, box_b.xmax)

    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w

    if inter_area == 0.0:
        return 0.0

    union_area = box_a.area + box_b.area - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def evaluate_item(
    gt: GroundTruthItem,
    pred: PredictionItem | None,
    iou_threshold: float = 0.5,
) -> EvalResult:
    """Evaluates a single prediction against its corresponding ground truth."""
    if pred is None:
        return EvalResult(
            item_id=gt.id,
            image_id=gt.image_id,
            label_match=False,
            iou=0.0,
            is_correct=False,
            status="MISSING_PREDICTION",
        )

    if pred.image_id != gt.image_id:
        return EvalResult(
            item_id=gt.id,
            image_id=gt.image_id,
            label_match=gt.label.strip().lower() == pred.label.strip().lower(),
            iou=0.0,
            is_correct=False,
            status="IMAGE_MISMATCH",
        )

    label_match = gt.label.strip().lower() == pred.label.strip().lower()
    iou = calculate_iou(gt.bbox, pred.bbox)
    is_correct = label_match and (iou >= iou_threshold)

    if is_correct:
        status = "CORRECT"
    elif not label_match:
        status = "CLASSIFICATION_ERROR"
    else:
        status = "LOCALIZATION_ERROR"

    return EvalResult(
        item_id=gt.id,
        image_id=gt.image_id,
        label_match=label_match,
        iou=round(iou, 4),
        is_correct=is_correct,
        status=status,
    )


def evaluate_batch(
    ground_truths: Sequence[GroundTruthItem],
    predictions: Sequence[PredictionItem],
    iou_threshold: float = 0.5,
) -> EvaluationSummary:
    """Evaluates a batch of ground truths and predictions."""
    pred_map = {p.id: p for p in predictions}
    results = [evaluate_item(gt, pred_map.get(gt.id), iou_threshold) for gt in ground_truths]

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = (correct / total) if total > 0 else 0.0
    mean_iou = (sum(r.iou for r in results) / total) if total > 0 else 0.0

    return EvaluationSummary(
        total_samples=total,
        correct_samples=correct,
        accuracy_at_threshold=round(accuracy, 4),
        mean_iou=round(mean_iou, 4),
        iou_threshold=iou_threshold,
        results=results,
    )
