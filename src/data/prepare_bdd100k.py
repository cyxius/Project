import yaml

import json
import logging
import shutil
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def _bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def prepare_splits(
    raw_dir: Path,
    processed_dir: Path,
    class_mapping: dict[str, int],
    clean_weather: list[str],
    adverse_weather: list[str],
    dsdl_train_json: Path,
    dsdl_val_json: Path,
    max_images: int | None = None,
) -> dict:
    images_base = raw_dir / "images" / "100k"

    splits = {
        "clean": {"images": 0, "boxes": 0},
        "adverse": {"images": 0, "boxes": 0},
    }

    # Determine which JSON files to process
    json_sources = []
    if dsdl_val_json.exists():
        json_sources.append(("val", dsdl_val_json, images_base / "val"))
    if dsdl_train_json.exists():
        json_sources.append(("train", dsdl_train_json, images_base / "train"))

    if not json_sources:
        raise FileNotFoundError(f"No DSDL sample JSON files found.")

    clean_anns, adverse_anns = [], []

    for subset, json_path, img_dir in json_sources:
        logger.info("Loading DSDL %s labels ...", subset)
        with open(json_path, "r") as f:
            data = json.load(f)

        samples = data.get("samples", [])
        for sample in samples:
            weather = sample.get("weather", "unknown").lower().strip()

            if weather in clean_weather:
                target = clean_anns
            elif weather in adverse_weather:
                target = adverse_anns
            else:
                continue

            media = sample.get("_media", {})
            img_name = media.get("name", "")
            if not img_name:
                continue

            img_shape = media.get("image_shape", [720, 1280])

            img_path = img_dir / img_name

            boxes = []
            for obj in sample.get("_objects", []):
                cat_name = obj.get("_category", "")
                if cat_name not in class_mapping:
                    continue

                coco_cls = class_mapping[cat_name]
                dsdl_bbox = obj.get("_bbox", [])
                if len(dsdl_bbox) != 4:
                    continue

                x1, y1, x2, y2 = _bbox_to_xyxy(dsdl_bbox)
                boxes.append({
                    "class_id": coco_cls,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "img_h": img_shape[0],
                    "img_w": img_shape[1],
                })

            if boxes:
                target.append({
                    "img_path": img_path,
                    "img_name": img_name,
                    "boxes": boxes,
                })

    logger.info("Parsed: %d clean, %d adverse annotated images.", len(clean_anns), len(adverse_anns))

    # Write each split
    for split_key, annotations in [("clean", clean_anns), ("adverse", adverse_anns)]:
        if max_images:
            annotations = annotations[:max_images]

        label_out = processed_dir / split_key / "labels"
        img_out = processed_dir / split_key / "images"
        label_out.mkdir(parents=True, exist_ok=True)
        img_out.mkdir(parents=True, exist_ok=True)

        total_boxes = 0
        copied = 0
        for ann in annotations:
            src = ann["img_path"]
            dst = img_out / ann["img_name"]
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
            elif not src.exists() and not dst.exists():
                continue

            first_box = ann["boxes"][0]
            img_w = first_box["img_w"]
            img_h = first_box["img_h"]

            with open(label_out / (Path(ann["img_name"]).stem + ".txt"), "w") as f:
                for box in ann["boxes"]:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    n_cx = cx / img_w
                    n_cy = cy / img_h
                    n_w = w / img_w
                    n_h = h / img_h
                    f.write(f"{box['class_id']} {n_cx:.6f} {n_cy:.6f} {n_w:.6f} {n_h:.6f}\n")
                    total_boxes += 1

        splits[split_key]["images"] = copied
        splits[split_key]["boxes"] = total_boxes
        logger.info("Wrote %s split: %d images, %d boxes.", split_key, copied, total_boxes)

    return splits


def generate_yolo_yaml(
    split_dir: Path,
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    nc = len(class_names)
    names_list = [class_names[k] for k in sorted(class_names.keys())]
    content = {
        "path": str(split_dir.resolve()),
        "train": "images",
        "val": "images",
        "nc": nc,
        "names": names_list,
    }
    with open(output_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    logger.info("Generated dataset YAML: %s", output_path)
