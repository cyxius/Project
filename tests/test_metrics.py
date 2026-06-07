import torch
from src.evaluation._matching import match_per_image


def test_perfect_match():
    """One prediction perfectly matches one GT."""
    pred_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    pred_classes = torch.tensor([0], dtype=torch.int64)
    pred_confs = torch.tensor([0.9], dtype=torch.float32)
    gt_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    gt_classes = torch.tensor([0], dtype=torch.int64)

    tp, confs, p_cls, num_gt = match_per_image(
        pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_threshold=0.5
    )
    assert tp == [1], f"Expected [1], got {tp}"
    assert num_gt == 1


def test_no_match_wrong_class():
    """Prediction and GT have different classes, no match."""
    pred_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    pred_classes = torch.tensor([0], dtype=torch.int64)
    pred_confs = torch.tensor([0.9], dtype=torch.float32)
    gt_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    gt_classes = torch.tensor([2], dtype=torch.int64)  # Different class

    tp, confs, p_cls, num_gt = match_per_image(
        pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_threshold=0.5
    )
    assert tp == [0], f"Expected [0], got {tp}"


def test_no_match_low_iou():
    """Prediction bbox far from GT, IoU below threshold."""
    pred_boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    pred_classes = torch.tensor([0], dtype=torch.int64)
    pred_confs = torch.tensor([0.9], dtype=torch.float32)
    gt_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    gt_classes = torch.tensor([0], dtype=torch.int64)

    tp, confs, p_cls, num_gt = match_per_image(
        pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_threshold=0.5
    )
    assert tp == [0], f"Expected [0], got {tp}"


def test_partial_overlap():
    pred_boxes = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
    pred_classes = torch.tensor([0], dtype=torch.int64)
    pred_confs = torch.tensor([0.9], dtype=torch.float32)
    gt_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    gt_classes = torch.tensor([0], dtype=torch.int64)

    tp, confs, p_cls, num_gt = match_per_image(
        pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_threshold=0.1
    )
    assert tp == [1], f"Expected [1], got {tp}"


def test_empty_gt():
    """No ground truth: all predictions are FP."""
    pred_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    pred_classes = torch.tensor([0], dtype=torch.int64)
    pred_confs = torch.tensor([0.9], dtype=torch.float32)
    gt_boxes = torch.empty((0, 4), dtype=torch.float32)
    gt_classes = torch.empty(0, dtype=torch.int64)

    tp, confs, p_cls, num_gt = match_per_image(
        pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_threshold=0.5
    )
    assert tp == [0]
    assert num_gt == 0


def test_empty_pred():
    """No predictions: returns empty lists."""
    pred_boxes = torch.empty((0, 4), dtype=torch.float32)
    pred_classes = torch.empty(0, dtype=torch.int64)
    pred_confs = torch.empty(0, dtype=torch.float32)
    gt_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    gt_classes = torch.tensor([0], dtype=torch.int64)

    tp, confs, p_cls, num_gt = match_per_image(
        pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_threshold=0.5
    )
    assert tp == []
    assert num_gt == 1


def test_multiple_predictions_one_gt():
    """Multiple predictions, only the best-matching one should be TP."""
    pred_boxes = torch.tensor([
        [100, 100, 200, 200],  # Perfect match -> TP
        [100, 100, 200, 200],  # Duplicate, GT already matched -> FP
    ], dtype=torch.float32)
    pred_classes = torch.tensor([0, 0], dtype=torch.int64)
    pred_confs = torch.tensor([0.9, 0.8], dtype=torch.float32)
    gt_boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    gt_classes = torch.tensor([0], dtype=torch.int64)

    tp, confs, p_cls, num_gt = match_per_image(
        pred_boxes, pred_classes, pred_confs, gt_boxes, gt_classes, iou_threshold=0.5
    )
    assert tp == [1, 0], f"Expected [1, 0], got {tp}"
    assert num_gt == 1


if __name__ == "__main__":
    test_perfect_match()
    test_no_match_wrong_class()
    test_no_match_low_iou()
    test_partial_overlap()
    test_empty_gt()
    test_empty_pred()
    test_multiple_predictions_one_gt()
    print("All IoU matching tests passed.")
