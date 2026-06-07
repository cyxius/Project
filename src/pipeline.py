
import logging
import json
from pathlib import Path

from .config import load_config
from .data.prepare_bdd100k import prepare_splits, generate_yolo_yaml
from .inference.predictor import predict_directory
from .evaluation.metrics import compute_map, save_metrics

logger = logging.getLogger(__name__)


def run_phase1(config_path: str, subset_size: int = None) -> dict:
    from ultralytics import YOLO
    config = load_config(config_path)
    out = config.output_dir
    logger.info("Preparing splits...")
    split_info = prepare_splits(
        raw_dir=config.dataset.raw_dir,
        processed_dir=config.dataset.processed_dir,
        class_mapping=config.dataset.class_mapping,
        clean_weather=config.dataset.clean_weather,
        adverse_weather=config.dataset.adverse_weather,
        dsdl_train_json=config.dataset.train_samples,
        dsdl_val_json=config.dataset.val_samples,
        max_images=subset_size,
    )

    for split_key in ["clean", "adverse"]:
        split_dir = config.dataset.processed_dir / split_key
        # Generate ultralytics YAML
        generate_yolo_yaml(
            split_dir,
            config.dataset.class_names,
            split_dir / f"data_{split_key}.yaml",
        )

    logger.info("Running inference...")
    model = YOLO(config.model.weights)

    all_summaries = {}
    # print(f"DEBUG: using {config.model.weights}")
    for split_key in ["clean", "adverse"]:
        img_dir = config.dataset.processed_dir / split_key / "images"
        pred_json = out / "predictions" / f"{split_key}_predictions.json"

        summary = predict_directory(
            model=model,
            image_dir=img_dir,
            output_json=pred_json,
            conf=config.model.conf_threshold,
            iou=config.model.iou_threshold,
            imgsz=config.model.imgsz,
            device=config.model.device,
            split_name=split_key,
        )
        all_summaries[split_key] = summary

    logger.info("Computing metrics...")
    all_metrics = {}
    for split_key in ["clean", "adverse"]:
        pred_json = out / "predictions" / f"{split_key}_predictions.json"
        label_dir = config.dataset.processed_dir / split_key / "labels"
        img_dir = config.dataset.processed_dir / split_key / "images"

        metrics = compute_map(
            predictions_json=pred_json,
            label_dir=label_dir,
            img_dir=img_dir,
            class_names=config.dataset.class_names,
            iou_threshold=config.evaluation.iou_threshold,
        )

        metrics_json = out / "metrics" / f"{split_key}_metrics.json"
        save_metrics(metrics, metrics_json)
        all_metrics[split_key] = metrics

    clean_metrics = all_metrics.get("clean", {})
    adverse_metrics = all_metrics.get("adverse", {})

    # save results
    run_meta = {
        "config_path": str(config_path),
        "subset_size": subset_size,
        "summaries": all_summaries,
        "clean_metrics": clean_metrics,
        "adverse_metrics": adverse_metrics,
    }
    with open(out / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    logger.info("Phase 1 done")
    return run_meta


def run_phase2(
    config_path: str,
    subset_size: int = None,
    skip_preprocessing: bool = False,
) -> dict:
    config = load_config(config_path)
    output_dir = config.phase2_output_dir
    for subdir in ["predictions", "metrics", "figures"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info("Checking data splits...")
    for split_key in ["clean", "adverse"]:
        split_dir = config.dataset.processed_dir / split_key
        if not (split_dir / "images").is_dir():
            logger.info("Data splits not found. Running data preparation...")
            prepare_splits(
                raw_dir=config.dataset.raw_dir,
                processed_dir=config.dataset.processed_dir,
                class_mapping=config.dataset.class_mapping,
                clean_weather=config.dataset.clean_weather,
                adverse_weather=config.dataset.adverse_weather,
                dsdl_train_json=config.dataset.train_samples,
                dsdl_val_json=config.dataset.val_samples,
                max_images=subset_size,
            )
            for sk in ["clean", "adverse"]:
                sdir = config.dataset.processed_dir / sk
                generate_yolo_yaml(sdir, config.dataset.class_names, sdir / f"data_{sk}.yaml")
            break

    logger.info("MSRCR preprocessing...")
    from .preprocessing import msrcr_directory

    adverse_img = config.dataset.processed_dir / "adverse" / "images"
    msrcr_out = config.dataset.processed_dir / "adverse_msrcr" / "images"
    if not skip_preprocessing or not msrcr_out.is_dir():
        msrcr_stats = msrcr_directory(
            adverse_img, msrcr_out,
            sigma_list=config.preprocessing.msrcr.sigma_list,
            alpha=config.preprocessing.msrcr.alpha,
            beta=config.preprocessing.msrcr.beta,
        )
    else:
        logger.info("Skipping MSRCR (--skip-preprocessing).")

    logger.info("CLAHE preprocessing...")
    from .preprocessing import enhance_directory

    clahe_out = config.dataset.processed_dir / "adverse_clahe" / "images"
    if not skip_preprocessing or not clahe_out.is_dir():
        clahe_stats = enhance_directory(
            adverse_img, clahe_out,
            clip_limit=config.preprocessing.clahe.clip_limit,
            tile_grid_size=config.preprocessing.clahe.tile_grid_size,
        )
    else:
        logger.info("Skipping CLAHE (--skip-preprocessing).")

    logger.info("Running inference...")
    model = YOLO(config.model.weights)

    splits = [("clean", config.dataset.processed_dir / "clean/images", config.dataset.processed_dir / "clean/labels"),
             ("adverse", config.dataset.processed_dir / "adverse/images", config.dataset.processed_dir / "adverse/labels"),
             ("adverse_msrcr", config.dataset.processed_dir / "adverse_msrcr/images", config.dataset.processed_dir / "adverse/labels"),
             ("adverse_clahe", config.dataset.processed_dir / "adverse_clahe/images", config.dataset.processed_dir / "adverse/labels")]

    for key, img, lbl in splits:
        pred_json = output_dir / "predictions" / f"{key}_predictions.json"
        predict_directory(
            model=model, image_dir=img, output_json=pred_json,
            conf=config.model.conf_threshold, iou=config.model.iou_threshold,
            imgsz=config.model.imgsz, device=config.model.device, split_name=key,
        )

    logger.info("Computing metrics...")
    all_metrics = {}
    for key, img, lbl in splits:
        pred_json = output_dir / "predictions" / f"{key}_predictions.json"
        metrics = compute_map(
            predictions_json=pred_json,
            label_dir=lbl,
            img_dir=img,
            class_names=config.dataset.class_names,
            iou_threshold=config.evaluation.iou_threshold,
        )
        save_metrics(metrics, output_dir / "metrics" / f"{key}_metrics.json")
        all_metrics[key] = metrics

    logger.info("Per-weather breakdown...")
    from .data.weather import build_weather_mapping, compute_weather_metrics

    weather_map = build_weather_mapping(
        config.dataset.train_samples,
        config.dataset.val_samples,
        config.dataset.processed_dir / "adverse" / "images",
    )
    weather_metrics = compute_weather_metrics(
        predictions_json=output_dir / "predictions" / "adverse_predictions.json",
        label_dir=config.dataset.processed_dir / "adverse" / "labels",
        img_dir=config.dataset.processed_dir / "adverse" / "images",
        weather_map=weather_map,
        class_names=config.dataset.class_names,
        iou_threshold=config.evaluation.iou_threshold,
    )

    # save results
    run_meta = {
        "config_path": str(config_path),
        "subset_size": subset_size,
        "metrics": {k: {"mAP": v.get("mAP")} for k, v in all_metrics.items()},
        "weather": {k: {"mAP": v.get("mAP"), "images": v.get("num_images")} for k, v in weather_metrics.items()},
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    logger.info("Phase 2 done")
    return {"all_metrics": all_metrics, "weather_metrics": weather_metrics}
