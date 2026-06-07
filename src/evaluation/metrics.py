
import json
import logging
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class

from ._matching import load_ground_truth_yolo, load_predictions_json, match_per_image

logger = logging.getLogger(__name__)


def compute_map(
    predictions_json: Path,
    label_dir: Path,
    img_dir: Path,
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
) -> dict:
    logger.info("Loading ground truth from %s ...", label_dir)
    gt = load_ground_truth_yolo(Path(label_dir), Path(img_dir))
    logger.info("Loading predictions from %s ...", predictions_json)
    preds = load_predictions_json(Path(predictions_json))

    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_gt_cls = []
    image_count = 0

    for img_name, gt_boxes_list in gt.items():
        image_count += 1
        pred_list = preds.get(img_name, [])

        gt_boxes = torch.tensor([b["bbox"] for b in gt_boxes_list], dtype=torch.float32)
        gt_classes = torch.tensor([b["class_id"] for b in gt_boxes_list], dtype=torch.int64)

        if pred_list:
            pred_list_sorted = sorted(pred_list, key=lambda x: x["confidence"], reverse=True)
            pred_boxes = torch.tensor([p["bbox"] for p in pred_list_sorted], dtype=torch.float32)
            pred_classes = torch.tensor([p["class_id"] for p in pred_list_sorted], dtype=torch.int64)
            pred_confs = torch.tensor([p["confidence"] for p in pred_list_sorted], dtype=torch.float32)

            tp, confs, p_cls, _ = match_per_image(
                pred_boxes, pred_classes, pred_confs,
                gt_boxes, gt_classes, iou_threshold,
            )
            all_tp.extend(tp)
            all_conf.extend(confs)
            all_pred_cls.extend(p_cls)

        if len(gt_classes) > 0:
            all_gt_cls.extend(gt_classes.tolist())

    if not all_tp:
        logger.warning("No predictions found for any image with ground truth.")
        return {
            "mAP": 0.0,
            "per_class_ap": {},
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_images": image_count,
            "num_predictions": 0,
            "num_gt_boxes": len(all_gt_cls),
        }

    tp_arr = np.array(all_tp, dtype=bool).reshape(-1, 1)
    conf_arr = np.array(all_conf, dtype=np.float32)
    pred_cls_arr = np.array(all_pred_cls, dtype=np.int64)
    target_cls = np.array(all_gt_cls, dtype=np.int64) if all_gt_cls else np.array([], dtype=np.int64)

    result = ap_per_class(
        tp_arr, conf_arr, pred_cls_arr, target_cls,
        names=class_names, plot=False, on_plot=None, save_dir=None,
    )

    _, _, p, r, f1, ap, ap_class, *_ = result

    per_class = {}
    valid_ap_list = []
    for i, cls_id in enumerate(ap_class):
        if cls_id not in class_names:
            continue
        ap_val = round(float(ap[i, 0]), 4)
        name = class_names[cls_id]
        per_class[name] = {
            "ap": ap_val,
            "precision": round(float(p[i]), 4) if i < len(p) else 0.0,
            "recall": round(float(r[i]), 4) if i < len(r) else 0.0,
            "f1": round(float(f1[i]), 4) if i < len(f1) else 0.0,
        }
        if ap_val > 0:
            valid_ap_list.append(ap_val)

    metrics = {
        "mAP": round(float(np.mean(valid_ap_list)) if valid_ap_list else 0.0, 4),
        "per_class_ap": per_class,
        "precision": round(float(np.mean(p[p > 0])) if (p > 0).any() else 0.0, 4),
        "recall": round(float(np.mean(r[r > 0])) if (r > 0).any() else 0.0, 4),
        "f1": round(float(np.mean(f1[f1 > 0])) if (f1 > 0).any() else 0.0, 4),
        "num_images": image_count,
        "num_predictions": len(all_tp),
        "num_gt_boxes": len(all_gt_cls),
    }

    logger.info(
        "Metrics: mAP@%.1f = %.4f | %d images, %d preds, %d GT boxes",
        iou_threshold, metrics["mAP"], image_count, metrics["num_predictions"], metrics["num_gt_boxes"],
    )
    return metrics


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics dict as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", output_path)


def load_metrics(json_path: Path) -> dict:
    """Load metrics from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)
