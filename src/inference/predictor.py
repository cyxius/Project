
import json
import logging
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def predict_directory(
    model: YOLO,
    image_dir: Path,
    output_json: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "cpu",
    split_name: str = "",
) -> dict:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = sorted(image_dir.glob("*.jpg"))
    if not image_files:
        logger.warning("No .jpg images found in %s", image_dir)
        return {"image_count": 0, "detection_count": 0, "avg_det_per_image": 0.0}

    label = f" [{split_name}]" if split_name else ""
    logger.info("Running inference on %d images%s ...", len(image_files), label)

    results_generator = model.predict(
        source=str(image_dir),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        stream=True,
        save=False,
        verbose=False,
    )

    predictions = []
    total_detections = 0

    for result in results_generator:
        image_name = Path(result.path).name
        detections = []

        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                cls_id = int(boxes.cls[i].item())
                conf_val = float(boxes.conf[i].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "bbox_xyxy": [round(v, 2) for v in xyxy],
                    "confidence": round(conf_val, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                })
                total_detections += 1

        predictions.append({
            "image_name": image_name,
            "detections": detections,
        })

    # Write JSON output
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(predictions, f, indent=2)

    image_count = len(predictions)
    avg_det = round(total_detections / image_count, 2) if image_count > 0 else 0.0

    summary = {
        "image_count": image_count,
        "detection_count": total_detections,
        "avg_det_per_image": avg_det,
    }
    logger.info("Inference done: %d images", image_count)
    return summary


def load_predictions(json_path: Path) -> list[dict]:
    """Load predictions from a JSON file produced by predict_directory."""
    with open(json_path, "r") as f:
        return json.load(f)
