import logging
import pprint

import awkward as ak
import yaml


def weighted_avg(pos, energy):
    weighted = pos * energy
    return ak.sum(weighted, axis=-1) / ak.sum(energy, axis=-1)


def load_config(config_path):
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_path=None, log_level="INFO"):
    """Set up and return a logger."""
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]

    if log_path:
        handlers.append(logging.FileHandler(log_path, mode="w"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def log_config(logger, config, header="Loaded Configuration"):
    logger.info(
        f"{header}:\n" + pprint.pformat(config, indent=2, compact=False)
    )
