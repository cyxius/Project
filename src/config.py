import yaml
from pathlib import Path
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> SimpleNamespace:
    """Load and validate YAML config, returning a namespace with absolute paths."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_root = path.parent.parent

    # Resolve path fields
    raw["dataset"]["raw_dir"] = str(project_root / raw["dataset"]["raw_dir"])
    raw["dataset"]["processed_dir"] = str(project_root / raw["dataset"]["processed_dir"])
    raw["dataset"]["dsdl_dir"] = str(project_root / raw["dataset"]["dsdl_dir"])
    raw["output_dir"] = str(project_root / raw["output_dir"])

    config = SimpleNamespace(
        dataset=SimpleNamespace(
            name=raw["dataset"]["name"],
            raw_dir=Path(raw["dataset"]["raw_dir"]),
            processed_dir=Path(raw["dataset"]["processed_dir"]),
            images_subdir=raw["dataset"]["images_subdir"],
            dsdl_dir=Path(raw["dataset"]["dsdl_dir"]),
            train_samples=Path(raw["dataset"]["dsdl_dir"]) / raw["dataset"]["train_samples"],
            val_samples=Path(raw["dataset"]["dsdl_dir"]) / raw["dataset"]["val_samples"],
            clean_weather=raw["dataset"]["clean_weather"],
            adverse_weather=raw["dataset"]["adverse_weather"],
            class_mapping=raw["dataset"]["class_mapping"],
            class_names=raw["dataset"]["class_names"],
        ),
        model=SimpleNamespace(
            weights=str(project_root / raw["model"]["weights"]),
            imgsz=raw["model"]["imgsz"],
            conf_threshold=raw["model"]["conf_threshold"],
            iou_threshold=raw["model"]["iou_threshold"],
            device=raw["model"]["device"],
            batch_size=raw["model"]["batch_size"],
        ),
        evaluation=SimpleNamespace(
            iou_threshold=raw["evaluation"]["iou_threshold"],
            max_det=raw["evaluation"]["max_det"],
        ),
        visualization=SimpleNamespace(
            dpi=raw["visualization"]["dpi"],
            fig_width=raw["visualization"]["fig_width"],
            fig_height=raw["visualization"]["fig_height"],
        ),
        preprocessing=SimpleNamespace(
            msrcr=SimpleNamespace(
                sigma_list=raw["preprocessing"]["msrcr"]["sigma_list"],
                alpha=raw["preprocessing"]["msrcr"]["alpha"],
                beta=raw["preprocessing"]["msrcr"]["beta"],
            ),
            clahe=SimpleNamespace(
                clip_limit=raw["preprocessing"]["clahe"]["clip_limit"],
                tile_grid_size=tuple(raw["preprocessing"]["clahe"]["tile_grid_size"]),
            ),
        ),
        seed=raw["seed"],
        output_dir=Path(raw["output_dir"]),
        phase2_output_dir=Path(raw["phase2_output_dir"]),
    )

    for subdir in ["predictions", "metrics", "figures"]:
        (config.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info("Configuration loaded successfully.")
    logger.info(f"  Dataset: %s", config.dataset.name)
    logger.info(f"  DSDL labels: %s", config.dataset.dsdl_dir)
    logger.info(f"  Model: %s", config.model.weights)
    logger.info(f"  Device: %s", config.model.device)
    logger.info(f"  Output dir: %s", config.output_dir)

    return config
