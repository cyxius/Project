from PIL import Image

import json
from pathlib import Path

import torch
import torchvision.ops as tv_ops


def load_ground_truth_yolo(label_dir: Path, img_dir: Path) -> dict:
    gt = {}
    label_files = sorted(label_dir.glob("*.txt"))

    for lf in label_files:
        img_name = lf.stem + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        boxes = []
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                n_cx, n_cy, n_w, n_h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                cx = n_cx * img_w
                cy = n_cy * img_h
                w = n_w * img_w
                h = n_h * img_h
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0

                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                })

        gt[img_name] = boxes

    return gt


def load_predictions_json(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)

    preds = {}
    for entry in data:
        img_name = entry["image_name"]
        detections = []
        for d in entry["detections"]:
            detections.append({
                "bbox": d["bbox_xyxy"],
                "confidence": d["confidence"],
                "class_id": d["class_id"],
            })
        preds[img_name] = detections

    return preds


def match_per_image(
    pred_boxes: torch.Tensor,
    pred_classes: torch.Tensor,
    pred_confs: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_threshold: float,
) -> tuple:
    # IoU matching, one GT per prediction (best match wins)
    if len(gt_boxes) == 0:
        return (
            [0] * len(pred_boxes),
            pred_confs.tolist(),
            pred_classes.tolist(),
            0,
        )

    if len(pred_boxes) == 0:
        return [], [], [], len(gt_boxes)

    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    tp, confs, p_cls = [], [], []

    for i in range(len(pred_boxes)):
        p_box = pred_boxes[i:i + 1]
        p_cls_val = pred_classes[i].item()
        p_conf = pred_confs[i].item()

        same_class = (gt_classes == p_cls_val) & (~gt_matched)

        if same_class.any():
            candidate_idx = torch.where(same_class)[0]
            candidate_gt = gt_boxes[candidate_idx]
            ious = tv_ops.box_iou(p_box, candidate_gt)[0]
            best_iou, best_local = ious.max(0)

            if best_iou >= iou_threshold:
                tp.append(1)
                gt_matched[candidate_idx[best_local]] = True
            else:
                tp.append(0)
        else:
            tp.append(0)

        confs.append(p_conf)
        p_cls.append(p_cls_val)

    return tp, confs, p_cls, len(gt_boxes)
