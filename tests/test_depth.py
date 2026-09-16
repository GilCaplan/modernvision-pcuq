"""Depth-map rendering: synthetic point clouds only, no downloads needed."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcuq.depth import render_depth_map


def test_shape_and_range():
    pts = torch.rand(200, 3) * 2 - 1  # uniform in [-1, 1]^3, unit-sphere-ish frame
    img = render_depth_map(pts, resolution=32, axis=2)
    assert img.shape == (32, 32)
    assert img.min() >= 0 and img.max() <= 1


def test_background_is_zero_where_no_point_lands():
    pts = torch.tensor([[0.0, 0.0, 0.5]])
    img = render_depth_map(pts, resolution=16, axis=2)
    assert (img > 0).sum() == 1
    assert img.max() > 0.5  # (0.5 + 1) / 2 = 0.75


def test_nearer_point_wins_the_pixel_regardless_of_order():
    # same (u, v), different depth along the view axis -> same pixel, contested
    near = torch.tensor([[0.1, 0.1, 0.9]])
    far = torch.tensor([[0.1, 0.1, -0.9]])
    img_a = render_depth_map(torch.cat([near, far]), resolution=8, axis=2)
    img_b = render_depth_map(torch.cat([far, near]), resolution=8, axis=2)
    assert torch.equal(img_a, img_b)
    assert img_a.max() > 0.9  # the near point's depth (0.9+1)/2 = 0.95 wins, not far's 0.05


def test_dilate_only_grows_the_lit_region():
    pts = torch.tensor([[0.0, 0.0, 0.5]])
    img0 = render_depth_map(pts, resolution=16, axis=2, dilate=0)
    img1 = render_depth_map(pts, resolution=16, axis=2, dilate=1)
    assert (img1 > 0).sum() > (img0 > 0).sum()
    lit = img0 > 0
    assert torch.equal(img1[lit], img0[lit])  # never overwrites real samples


def test_axis_picks_the_projection_plane():
    # a line of points along x only varies along the view axis when axis=0
    pts = torch.stack([torch.linspace(-1, 1, 50),
                       torch.zeros(50), torch.zeros(50)], dim=1)
    along_x = render_depth_map(pts, resolution=16, axis=0)  # collapses to one column
    along_z = render_depth_map(pts, resolution=16, axis=2)  # spreads across a row
    assert (along_x > 0).sum() == 1
    assert (along_z > 0).sum() > 1
