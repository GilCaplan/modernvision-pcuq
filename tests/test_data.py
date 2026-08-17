"""OFF parsing, mesh sampling, and normalization — the ModelNet path without the
2GB download."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcuq.data import _parse_off, normalize_unit_sphere, sample_mesh

TETRA = """OFF
4 4 0
0 0 0
1 0 0
0 1 0
0 0 1
3 0 1 2
3 0 1 3
3 0 2 3
3 1 2 3
"""

# ModelNet40's malformed variant: header glued to the counts, and a quad face.
TETRA_GLUED = TETRA.replace("OFF\n4 4 0", "OFF4 4 0").replace(
    "3 1 2 3", "4 1 2 3 0")


@pytest.mark.parametrize("text", [TETRA, TETRA_GLUED])
def test_parse_off(text):
    verts, faces = _parse_off(text)
    assert verts.shape == (4, 3)
    assert faces.shape[1] == 3 and faces.shape[0] >= 4  # quad fan-triangulated
    assert faces.max() < 4


def test_sample_mesh_deterministic_and_on_surface():
    verts, faces = _parse_off(TETRA)
    p1 = sample_mesh(verts, faces, 128, seed=7)
    p2 = sample_mesh(verts, faces, 128, seed=7)
    assert torch.equal(p1, p2)
    assert p1.shape == (128, 3)
    # Tetra surface points satisfy x,y,z >= 0 and x+y+z <= 1.
    assert (p1 >= -1e-9).all() and (p1.sum(dim=1) <= 1 + 1e-9).all()


def test_normalize_unit_sphere():
    pts = torch.randn(64, 3, dtype=torch.float64) * 5 + 3
    out, center, scale = normalize_unit_sphere(pts)
    assert torch.allclose(out.mean(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-9)
    assert torch.isclose(out.norm(dim=1).max(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(out * scale + center, pts)
