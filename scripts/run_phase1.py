#!/usr/bin/env python3
"""Phase 1 (Baseline) and Phase 2 (Preprocessing Recovery) CLI entry point."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Weather-Robustness Evaluation Pipeline"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=1,
        help="Which phase to run: 1=Baseline, 2=Preprocessing Recovery.",
    )
    parser.add_argument(
        "--config", type=str, default="config/phase1_config.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip BDD100K dataset verification.",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Number of images per split for quick testing.",
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="(Phase 2 only) Skip MSRCR/CLAHE if already processed.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.phase == 2:
        from src.pipeline import run_phase2
        run_phase2(
            config_path=args.config,
            skip_download=args.skip_download,
            subset_size=args.subset,
            skip_preprocessing=args.skip_preprocessing,
        )
    else:
        from src.pipeline import run_phase1
        run_phase1(
            config_path=args.config,
            skip_download=args.skip_download,
            subset_size=args.subset,
        )


if __name__ == "__main__":
    main()
