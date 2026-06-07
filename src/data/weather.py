from collections import Counter

import json
import logging
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class

from ..evaluation._matching import load_ground_truth_yolo, load_predictions_json, match_per_image

logger = logging.getLogger(__name__)


def build_weather_mapping(
    dsdl_train_json: Path,
    dsdl_val_json: Path,
    image_dir: Path,
) -> dict[str, str]:
    weather_map = {}
    existing = set(p.name for p in image_dir.glob("*.jpg"))

    for json_path in [dsdl_train_json, dsdl_val_json]:
        if not json_path.exists():
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        for sample in data.get("samples", []):
            name = sample.get("_media", {}).get("name", "")
            weather = sample.get("weather", "")
            if name and weather and name in existing:
                weather_map[name] = weather.lower().strip()

    for w, c in Counter(weather_map.values()).most_common():
        logger.info("  Weather '%s': %d images", w, c)

    return weather_map


def compute_weather_metrics(
    predictions_json: Path,
    label_dir: Path,
    img_dir: Path,
    weather_map: dict[str, str],
    class_names: dict[int, str],
    iou_threshold: float = 0.5,
    target_weathers: list[str] | None = None,
) -> dict[str, dict]:
    gt = load_ground_truth_yolo(Path(label_dir), Path(img_dir))
    preds = load_predictions_json(Path(predictions_json))

    weather_groups = {}
    if target_weathers is None:
        weathers_seen = set(weather_map.values())
        target_weathers = sorted(weathers_seen)

    for w in target_weathers:
        weather_groups[w] = {"images": [], "tp": [], "conf": [], "pred_cls": [], "gt_cls": []}

    for img_name, gt_boxes_list in gt.items():
        w = weather_map.get(img_name, "unknown")
        if w not in weather_groups:
            continue
        weather_groups[w]["images"].append(img_name)

        pred_list = preds.get(img_name, [])

        gt_boxes = torch.tensor([b["bbox"] for b in gt_boxes_list], dtype=torch.float32)
        gt_classes = torch.tensor([b["class_id"] for b in gt_boxes_list], dtype=torch.int64)

        if len(gt_classes) > 0:
            weather_groups[w]["gt_cls"].extend(gt_classes.tolist())

        if pred_list:
            pred_list_sorted = sorted(pred_list, key=lambda x: x["confidence"], reverse=True)
            p_boxes = torch.tensor([p["bbox"] for p in pred_list_sorted], dtype=torch.float32)
            p_classes = torch.tensor([p["class_id"] for p in pred_list_sorted], dtype=torch.int64)
            p_confs = torch.tensor([p["confidence"] for p in pred_list_sorted], dtype=torch.float32)

            tp, confs, pred_cls, _ = match_per_image(
                p_boxes, p_classes, p_confs, gt_boxes, gt_classes, iou_threshold,
            )
            weather_groups[w]["tp"].extend(tp)
            weather_groups[w]["conf"].extend(confs)
            weather_groups[w]["pred_cls"].extend(pred_cls)

    results = {}
    for w in target_weathers:
        g = weather_groups[w]
        if not g["tp"]:
            results[w] = {"mAP": 0.0, "per_class_ap": {}, "num_images": len(g["images"]),
                          "num_predictions": 0, "num_gt_boxes": len(g["gt_cls"])}
            continue

        tp_arr = np.array(g["tp"], dtype=bool).reshape(-1, 1)
        conf_arr = np.array(g["conf"], dtype=np.float32)
        pred_cls_arr = np.array(g["pred_cls"], dtype=np.int64)
        target_cls = np.array(g["gt_cls"], dtype=np.int64)

        result = ap_per_class(
            tp_arr, conf_arr, pred_cls_arr, target_cls,
            names=class_names, plot=False, on_plot=None, save_dir=None,
        )
        _, _, p, r, f1, ap, ap_class, *_ = result

        per_class = {}
        valid_ap = []
        for i, cls_id in enumerate(ap_class):
            if cls_id not in class_names:
                continue
            ap_val = round(float(ap[i, 0]), 4)
            name = class_names[cls_id]
            per_class[name] = {
                "ap": ap_val,
                "precision": round(float(p[i]), 4) if i < len(p) else 0.0,
                "recall": round(float(r[i]), 4) if i < len(r) else 0.0,
            }
            if ap_val > 0:
                valid_ap.append(ap_val)

        results[w] = {
            "mAP": round(float(np.mean(valid_ap)) if valid_ap else 0.0, 4),
            "per_class_ap": per_class,
            "num_images": len(g["images"]),
            "num_predictions": len(g["tp"]),
            "num_gt_boxes": len(g["gt_cls"]),
        }

    return results
