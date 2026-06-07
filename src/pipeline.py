"""Phase 1 orchestrator: dataset prep -> inference -> metrics -> visualization."""

import logging
import json
from pathlib import Path

from .config import load_config
from .data.prepare_bdd100k import prepare_splits, generate_yolo_yaml
from .inference.predictor import predict_directory
from .evaluation.metrics import compute_map, save_metrics
try:
    from .data.download import print_guide, verify_dataset
    from .data.dataset import get_split_stats, verify_label_integrity, print_stats
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False
try:
    from .visualization.charts import (
        plot_map_comparison, plot_pr_curves, plot_confidence_histogram,
        plot_map_comparison_4way, plot_recovery_bars, plot_pr_curves_4way, plot_confidence_histogram_4way,
    )
    from .visualization.report import print_summary, print_summary_4way
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

logger = logging.getLogger(__name__)


def run_phase1(config_path: str, skip_download: bool = False, subset_size: int = None) -> dict:
    """
    Execute the full Phase 1 pipeline.

    Args:
        config_path: Path to config/phase1_config.yaml
        skip_download: If True, assume BDD100K is already downloaded.
        subset_size: If set, only process this many images per split (for testing).

    Returns:
        dict with clean_metrics, foggy_metrics, and output paths.
    """
    config = load_config(config_path)

    output_dir = config.output_dir

    # ---- Step 0: Dataset verification or guide ----
    if _HAS_UTILS and not skip_download:
        stats = verify_dataset(config.dataset.raw_dir)
        if not stats["ready"]:
            print_guide(config.dataset.raw_dir)
            logger.warning("BDD100K dataset not ready. Follow instructions above and re-run.")
            return {}
    elif not _HAS_UTILS:
        logger.info("Skipping dataset verification (module not available).")
    else:
        logger.info("Skipping dataset verification (--skip-download).")

    # ---- Step 1: Prepare clean/foggy splits ----
    logger.info("=== Step 1: Preparing dataset splits ===")
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
        if _HAS_UTILS:
            stats = get_split_stats(split_dir, config.dataset.class_names)
            print_stats(stats, split_key)
            integrity = verify_label_integrity(split_dir)
            if not integrity["ok"]:
                logger.warning("Label integrity issues in %s split.", split_key)

    # ---- Step 2: Batch inference ----
    logger.info("=== Step 2: Running batch inference ===")
    from ultralytics import YOLO

    model = YOLO(config.model.weights)

    all_summaries = {}
    for split_key in ["clean", "adverse"]:
        img_dir = config.dataset.processed_dir / split_key / "images"
        pred_json = output_dir / "predictions" / f"{split_key}_predictions.json"

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

    # ---- Step 3: Compute evaluation metrics ----
    logger.info("=== Step 3: Computing evaluation metrics ===")
    all_metrics = {}
    for split_key in ["clean", "adverse"]:
        pred_json = output_dir / "predictions" / f"{split_key}_predictions.json"
        label_dir = config.dataset.processed_dir / split_key / "labels"
        img_dir = config.dataset.processed_dir / split_key / "images"

        metrics = compute_map(
            predictions_json=pred_json,
            label_dir=label_dir,
            img_dir=img_dir,
            class_names=config.dataset.class_names,
            iou_threshold=config.evaluation.iou_threshold,
        )

        metrics_json = output_dir / "metrics" / f"{split_key}_metrics.json"
        save_metrics(metrics, metrics_json)
        all_metrics[split_key] = metrics

    # ---- Step 4: Visualization ----
    logger.info("=== Step 4: Generating visualizations ===")
    figures_dir = output_dir / "figures"

    clean_metrics = all_metrics.get("clean", {})
    foggy_metrics = all_metrics.get("adverse", {})

    if _HAS_VIZ and clean_metrics and foggy_metrics:
        plot_map_comparison(clean_metrics, foggy_metrics, figures_dir, dpi=config.visualization.dpi)
        plot_pr_curves(clean_metrics, foggy_metrics, figures_dir, dpi=config.visualization.dpi)
        plot_confidence_histogram(clean_metrics, foggy_metrics, figures_dir, dpi=config.visualization.dpi)

        print_summary(clean_metrics, foggy_metrics)
    elif not clean_metrics or not foggy_metrics:
        logger.warning("Skipping visualization: missing metrics for one or both splits.")

    # ---- Save run metadata ----
    run_meta = {
        "config_path": str(config_path),
        "subset_size": subset_size,
        "summaries": all_summaries,
        "clean_metrics": clean_metrics,
        "foggy_metrics": foggy_metrics,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    logger.info("=== Phase 1 pipeline complete. Outputs in %s ===", output_dir)
    return run_meta


def run_phase2(
    config_path: str,
    skip_download: bool = False,
    subset_size: int = None,
    skip_preprocessing: bool = False,
) -> dict:
    """
    Execute Phase 2: apply MSRCR + CLAHE preprocessing, re-run inference,
    and quantify detection accuracy recovery.

    Args:
        config_path: Path to config YAML.
        skip_download: Skip dataset verification.
        subset_size: Max images per split (testing).
        skip_preprocessing: If True, skip MSRCR/CLAHE (use existing output dirs).

    Returns:
        dict with all_metrics, weather_metrics, and output paths.
    """
    config = load_config(config_path)
    output_dir = config.phase2_output_dir
    for subdir in ["predictions", "metrics", "figures"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ---- Step 0: Ensure data splits exist ----
    logger.info("=== Step 0: Checking data splits ===")
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

    # ---- Step 1: MSRCR Enhancement ----
    logger.info("=== Step 1: MSRCR Enhancement ===")
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

    # ---- Step 2: CLAHE Enhancement ----
    logger.info("=== Step 2: CLAHE Enhancement ===")
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

    # ---- Step 3: Batch inference on 4 splits ----
    logger.info("=== Step 3: Running batch inference ===")
    from ultralytics import YOLO

    model = YOLO(config.model.weights)

    SPLITS = [
        {"key": "clean",         "img": config.dataset.processed_dir / "clean/images",           "lbl": config.dataset.processed_dir / "clean/labels"},
        {"key": "adverse",       "img": config.dataset.processed_dir / "adverse/images",         "lbl": config.dataset.processed_dir / "adverse/labels"},
        {"key": "adverse_msrcr",   "img": config.dataset.processed_dir / "adverse_msrcr/images",     "lbl": config.dataset.processed_dir / "adverse/labels"},
        {"key": "adverse_clahe", "img": config.dataset.processed_dir / "adverse_clahe/images",   "lbl": config.dataset.processed_dir / "adverse/labels"},
    ]

    for sp in SPLITS:
        pred_json = output_dir / "predictions" / f"{sp['key']}_predictions.json"
        predict_directory(
            model=model, image_dir=sp["img"], output_json=pred_json,
            conf=config.model.conf_threshold, iou=config.model.iou_threshold,
            imgsz=config.model.imgsz, device=config.model.device, split_name=sp["key"],
        )

    # ---- Step 4: Compute metrics for all 4 splits ----
    logger.info("=== Step 4: Computing evaluation metrics ===")
    all_metrics = {}
    for sp in SPLITS:
        pred_json = output_dir / "predictions" / f"{sp['key']}_predictions.json"
        metrics = compute_map(
            predictions_json=pred_json,
            label_dir=sp["lbl"],
            img_dir=sp["img"],
            class_names=config.dataset.class_names,
            iou_threshold=config.evaluation.iou_threshold,
        )
        save_metrics(metrics, output_dir / "metrics" / f"{sp['key']}_metrics.json")
        all_metrics[sp["key"]] = metrics

    # ---- Step 5: Per-weather breakdown ----
    logger.info("=== Step 5: Per-weather breakdown ===")
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

    # ---- Step 6: Visualization ----
    logger.info("=== Step 6: Generating visualizations ===")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if _HAS_VIZ:
        plot_map_comparison_4way(all_metrics, figures_dir, dpi=config.visualization.dpi)
        plot_recovery_bars(all_metrics, figures_dir, dpi=config.visualization.dpi)
        plot_pr_curves_4way(all_metrics, figures_dir, dpi=config.visualization.dpi)
        plot_confidence_histogram_4way(all_metrics, figures_dir, dpi=config.visualization.dpi)

        # ---- Step 7: Report ----
        print_summary_4way(all_metrics, weather_metrics)

    # ---- Save run metadata ----
    run_meta = {
        "config_path": str(config_path),
        "subset_size": subset_size,
        "metrics": {k: {"mAP": v.get("mAP")} for k, v in all_metrics.items()},
        "weather": {k: {"mAP": v.get("mAP"), "images": v.get("num_images")} for k, v in weather_metrics.items()},
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2, default=str)

    logger.info("=== Phase 2 pipeline complete. Outputs in %s ===", output_dir)
    return {"all_metrics": all_metrics, "weather_metrics": weather_metrics}
