from __future__ import annotations

import numpy as np
import pytest
from skimage.measure import marching_cubes

from vdata_standalone.reconstruction import (
    ReconstructionError, box_fraction, reconstruct_fraction_distance, pad_distance_field,
    solve_plane_offsets,
)


@pytest.mark.parametrize("normal,offset,expected", [
    ((1., 0., 0.), -.25, .25), ((-1., 0., 0.), -.25, .25),
    ((1., 1., 0.), -.5, .125), ((1., 1., 1.), -1., 1/48),
    ((1., 1., 1.), 0., .5), ((1., 1., 1.), 1., 47/48),
    ((1., 1e-9, 1e-12), .1, .6),
])
def test_box_fraction_known_geometry(normal, offset, expected):
    assert float(box_fraction(normal, offset)) == pytest.approx(expected, abs=2e-12)


def test_anisotropy_normal_scaling_complement_and_inverse():
    rng = np.random.default_rng(20261026)
    n = rng.normal(size=(200, 3))
    c = rng.uniform(.001, .999, size=200)
    h = np.array([.4, 1.1, 2.3])
    d = solve_plane_offsets(c, n, h)
    assert np.max(np.abs(box_fraction(n, d, h) - c)) < 2e-9
    assert np.max(np.abs(box_fraction(n, -d, h) - (1-c))) < 2e-9
    assert np.max(np.abs(box_fraction(n*3, d*3, h) - c)) < 2e-9
    np.testing.assert_array_equal(solve_plane_offsets(np.full(len(n), .5), n, h), np.zeros(len(n)))


def _plane_field(normal, spacing=(1, 1, 1), shape=(22, 24, 26), quantize=False):
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    h = np.asarray(spacing, dtype=float)
    centers = (np.moveaxis(np.indices(shape, dtype=float), 0, -1) + .5) * h
    alpha = np.dot(n, np.asarray(shape)*h/2) + .23
    fraction = box_fraction(n, alpha - centers@n, h)
    if quantize:
        fraction = np.rint(fraction*128)/128
    return fraction, n, h, alpha


def _errors(field, n, h, alpha):
    vertices, _, _, _ = marching_cubes(field.astype(np.float32), level=0, spacing=tuple(h))
    vertices += h*.5
    index = vertices/h
    interior = np.all((index > 4) & (index < np.asarray(field.shape)-4), axis=1)
    return np.abs(vertices[interior]@n-alpha)


@pytest.mark.parametrize("normal", [(1, 0, 0), (1, .23, .37), (1, 1, 1), (-.73, 1, .2)])
def test_supplied_planar_normals_recover_interface(normal):
    c, n, h, alpha = _plane_field(normal)
    before = c.copy()
    field, report = reconstruct_fraction_distance(c, h, normals_xyz=n)
    assert np.array_equal(c, before)
    assert _errors(field, n, h, alpha).max() < 3e-6
    assert report["max_local_fraction_residual"] < 1e-8
    assert report["plane_equations_used"] is False
    assert report["final_mc_volume_conservation_guaranteed"] is False


@pytest.mark.parametrize("normal", [(1, 0, 0), (1, .23, .37), (1, 1, 1), (-.73, 1, .2)])
def test_estimated_normals_improve_quantized_planar_interior(normal):
    c, n, h, alpha = _plane_field(normal, quantize=True)
    baseline = _errors(c-.5, n, h, alpha)
    field, report = reconstruct_fraction_distance(c, h)
    error = _errors(field, n, h, alpha)
    assert np.sqrt(np.mean(error**2)) < .2*np.sqrt(np.mean(baseline**2))
    assert error.max() < .02
    assert report["calibrated_mixed_cells"] == np.count_nonzero((c>0)&(c<1))


def test_anisotropic_known_plane():
    c, n, h, alpha = _plane_field((1, -.3, .7), spacing=(1, 2, .6))
    field, _ = reconstruct_fraction_distance(c, h, normals_xyz=n)
    assert _errors(field, n, h, alpha).max() < 5e-6


def test_no_mixed_cells_and_undefined_normals_do_not_invent_direction():
    binary = np.zeros((6, 6, 6), dtype=np.float32)
    binary[2:4, 2:4, 2:4] = 1
    field, report = reconstruct_fraction_distance(binary, (1, 1, 1))
    np.testing.assert_array_equal(field, binary-.5)
    assert report["mixed_cells"] == 0
    uniform = np.full((4, 4, 4), .7, dtype=np.float32)
    field, report = reconstruct_fraction_distance(uniform, (1, 1, 1))
    np.testing.assert_array_equal(field, uniform-.5)
    assert report["undefined_normal_cells"] == uniform.size
    assert report["calibrated_mixed_cells"] == 0


def test_chunk_invariance_and_pure_sign_preservation():
    c, _, h, _ = _plane_field((1, .23, .37), quantize=True)
    field, report = reconstruct_fraction_distance(c, h, chunk_size=17)
    full, _ = reconstruct_fraction_distance(c, h, chunk_size=100000)
    np.testing.assert_array_equal(field, full)
    assert np.all(field[c==0] < 0)
    assert np.all(field[c==1] > 0)


@pytest.mark.parametrize("c,h", [
    (np.zeros((3,3)), (1,1,1)), (np.full((3,3,3), np.nan), (1,1,1)),
    (np.full((3,3,3), 1.2), (1,1,1)), (np.zeros((3,3,3)), (1,0,1)),
])
def test_invalid_fraction_input(c, h):
    with pytest.raises(ReconstructionError):
        reconstruct_fraction_distance(c, h)


def test_invalid_plane_input():
    with pytest.raises(ReconstructionError):
        box_fraction((0,0,0), 0)
    with pytest.raises(ReconstructionError):
        solve_plane_offsets(1.1, (1,0,0))
    with pytest.raises(ReconstructionError):
        reconstruct_fraction_distance(np.full((3,3,3), .5), (1,1,1), normals_xyz=(0,0,0))


def test_reflected_ghost_closes_on_domain_face_despite_nonuniform_distances():
    rng = np.random.default_rng(12)
    field = rng.uniform(.1, 2., size=(5, 6, 7)).astype(np.float32)
    padded = pad_distance_field(field)
    vertices, _, _, _ = marching_cubes(padded, 0, allow_degenerate=False)
    vertices -= .5
    np.testing.assert_allclose(vertices.min(axis=0), 0, atol=2e-7)
    np.testing.assert_allclose(vertices.max(axis=0), field.shape, atol=2e-7)
    np.testing.assert_array_equal(padded[1:-1,1:-1,1:-1], field)
