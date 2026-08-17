"""CPU-fast sanity tests. Run: python -m pytest tests/"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcuq.data import corrupt
from pcuq.utils import apply_overrides, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_config_profiles_share_schema():
    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out.add(prefix + k)
            if isinstance(v, dict):
                out |= keys(v, prefix + k + ".")
        return out

    local = load_config(CONFIGS / "local.yaml")
    gpu = load_config(CONFIGS / "gpu.yaml")
    assert keys(local) == keys(gpu), "local.yaml and gpu.yaml drifted apart"


def test_override_parsing():
    cfg = load_config(CONFIGS / "local.yaml")
    cfg = apply_overrides(cfg, ["spectrum.n_ev=7", "denoiser.kind=noise2score3d"])
    assert cfg["spectrum"]["n_ev"] == 7
    assert cfg["denoiser"]["kind"] == "noise2score3d"


def test_corrupt_is_seeded_and_index_preserving():
    x = torch.randn(2, 64, 3)
    y1 = corrupt(x, sigma=0.05, seed=123)
    y2 = corrupt(x, sigma=0.05, seed=123)
    assert torch.equal(y1, y2), "same seed must give same corruption"
    assert y1.shape == x.shape
    assert torch.allclose((y1 - x).std(), torch.tensor(0.05), atol=0.005)
