import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from projects.vlm_grounding_evaluator.models import BoundingBox, GroundTruthItem, PredictionItem
    from projects.vlm_grounding_evaluator.evaluator import calculate_iou, evaluate_item, evaluate_batch
    from projects.vlm_grounding_evaluator.main import run_evaluation
except ImportError:
    from models import BoundingBox, GroundTruthItem, PredictionItem
    from evaluator import calculate_iou, evaluate_item, evaluate_batch
    from main import run_evaluation


def test_bounding_box_validation():
    # Valid box
    box = BoundingBox(ymin=0.1, xmin=0.2, ymax=0.5, xmax=0.6)
    assert box.area == pytest.approx(0.4 * 0.4)

    # Inverted y coordinates
    with pytest.raises(ValidationError):
        BoundingBox(ymin=0.6, xmin=0.1, ymax=0.2, xmax=0.5)

    # Inverted x coordinates
    with pytest.raises(ValidationError):
        BoundingBox(ymin=0.1, xmin=0.8, ymax=0.5, xmax=0.3)

    # Coordinates out of bounds (> 1.0 or < 0.0)
    with pytest.raises(ValidationError):
        BoundingBox(ymin=-0.1, xmin=0.0, ymax=0.5, xmax=0.5)
    with pytest.raises(ValidationError):
        BoundingBox(ymin=0.0, xmin=0.0, ymax=1.2, xmax=0.5)


def test_iou_calculation():
    # Exact overlap
    b1 = BoundingBox(ymin=0.2, xmin=0.2, ymax=0.6, xmax=0.6)
    assert calculate_iou(b1, b1) == pytest.approx(1.0)

    # Completely disjoint
    b2 = BoundingBox(ymin=0.7, xmin=0.7, ymax=0.9, xmax=0.9)
    assert calculate_iou(b1, b2) == pytest.approx(0.0)

    # Edge touching
    b3 = BoundingBox(ymin=0.6, xmin=0.2, ymax=0.8, xmax=0.6)
    assert calculate_iou(b1, b3) == pytest.approx(0.0)

    # Half overlap along x
    box_left = BoundingBox(ymin=0.0, xmin=0.0, ymax=1.0, xmax=0.5)
    box_right = BoundingBox(ymin=0.0, xmin=0.25, ymax=1.0, xmax=0.75)
    assert calculate_iou(box_left, box_right) == pytest.approx(1.0 / 3.0)


def test_evaluate_item_statuses():
    gt = GroundTruthItem(
        id="1",
        image_id="img_1",
        label="dog",
        bbox=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5),
    )

    # Missing prediction
    res_missing = evaluate_item(gt, None)
    assert res_missing.status == "MISSING_PREDICTION"
    assert not res_missing.is_correct

    # Correct prediction (IoU >= 0.5 and label matches)
    pred_correct = PredictionItem(
        id="1",
        image_id="img_1",
        label="dog",
        bbox=BoundingBox(ymin=0.12, xmin=0.12, ymax=0.5, xmax=0.5),
    )
    res_correct = evaluate_item(gt, pred_correct, iou_threshold=0.5)
    assert res_correct.status == "CORRECT"
    assert res_correct.is_correct

    # Localization error (label matches, but IoU < 0.5)
    pred_loc_err = PredictionItem(
        id="1",
        image_id="img_1",
        label="dog",
        bbox=BoundingBox(ymin=0.4, xmin=0.4, ymax=0.7, xmax=0.7),
    )
    res_loc = evaluate_item(gt, pred_loc_err, iou_threshold=0.5)
    assert res_loc.status == "LOCALIZATION_ERROR"
    assert not res_loc.is_correct

    # Classification error (label mismatch)
    pred_cls_err = PredictionItem(
        id="1",
        image_id="img_1",
        label="cat",
        bbox=BoundingBox(ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.5),
    )
    res_cls = evaluate_item(gt, pred_cls_err, iou_threshold=0.5)
    assert res_cls.status == "CLASSIFICATION_ERROR"
    assert not res_cls.is_correct


def test_evaluate_batch_summary():
    gt1 = GroundTruthItem(id="1", image_id="img_1", label="car", bbox=BoundingBox(ymin=0.0, xmin=0.0, ymax=0.5, xmax=0.5))
    gt2 = GroundTruthItem(id="2", image_id="img_1", label="bike", bbox=BoundingBox(ymin=0.5, xmin=0.5, ymax=1.0, xmax=1.0))

    p1 = PredictionItem(id="1", image_id="img_1", label="car", bbox=BoundingBox(ymin=0.0, xmin=0.0, ymax=0.5, xmax=0.5))

    summary = evaluate_batch([gt1, gt2], [p1], iou_threshold=0.5)
    assert summary.total_samples == 2
    assert summary.correct_samples == 1
    assert summary.accuracy_at_threshold == 0.5
    assert summary.mean_iou == 0.5


def test_main_runner_on_sample_dataset():
    sample_path = CURRENT_DIR / "sample_dataset.json"
    summary = run_evaluation(sample_path, iou_threshold=0.5)
    assert summary.total_samples == 5
    assert summary.correct_samples == 2
    assert summary.accuracy_at_threshold == 0.4
