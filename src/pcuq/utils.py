"""Config loading, seeding, device selection, output dirs."""

import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """Load a YAML profile (configs/local.yaml or configs/gpu.yaml) into a dict.

    Kept as a plain nested dict on purpose: the schema is documented in the YAMLs
    themselves and in docs/WORKFLOW.md.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply CLI overrides like 'spectrum.n_ev=3' onto a loaded config."""
    for ov in overrides:
        key, _, raw = ov.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        try:
            node[leaf] = yaml.safe_load(raw)
        except yaml.YAMLError:
            node[leaf] = raw
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want != "auto":
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_out_dir(cfg: dict, experiment: str) -> Path:
    """outputs/<profile-name>/<experiment>/, with the resolved config dumped inside."""
    out = Path(cfg.get("out_dir", "outputs")) / cfg["name"] / experiment
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return out
