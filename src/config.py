import yaml
from pathlib import Path
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)


def _dict_to_ns(d):
    """Recursively convert dict to SimpleNamespace, resolving Path fields."""
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _dict_to_ns(v)
    return SimpleNamespace(**d)


def load_config(config_path: str) -> SimpleNamespace:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_root = path.parent.parent
    dsdl = Path(raw["dataset"]["dsdl_dir"])

    raw["dataset"]["raw_dir"] = project_root / raw["dataset"]["raw_dir"]
    raw["dataset"]["processed_dir"] = project_root / raw["dataset"]["processed_dir"]
    raw["dataset"]["dsdl_dir"] = dsdl
    raw["dataset"]["train_samples"] = dsdl / raw["dataset"]["train_samples"]
    raw["dataset"]["val_samples"] = dsdl / raw["dataset"]["val_samples"]
    raw["output_dir"] = project_root / raw["output_dir"]
    raw["phase2_output_dir"] = project_root / raw["phase2_output_dir"]
    raw["model"]["weights"] = str(project_root / raw["model"]["weights"])

    config = _dict_to_ns(raw)

    for subdir in ["predictions", "metrics", "figures"]:
        (config.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    return config
