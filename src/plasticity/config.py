"""Load simulation settings from YAML."""

from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "2004.yaml"


def load_config(config_path=DEFAULT_CONFIG):
    """Read the original settings, including YAML infinity and the output path."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config["EVOLUTION_EQUATION"] not in (5, 6, 7):
        raise ValueError("EVOLUTION_EQUATION must be 5, 6, or 7.")

    config["DESKTOP_PATH"] = Path(config["DESKTOP_PATH"]).expanduser()
    return config
