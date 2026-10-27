"""Active-region masks must retain the unmasked Lewiner geometry exactly."""

import numpy as np
import pytest
from skimage.measure import marching_cubes

from vdata_standalone.meshing import MeshingError, _active_block_mask, mesh_material


def encoded(fraction, *, origin=(8e-3, -2e-2, 3e-4), spacing=20e-9):
    raw = np.empty(fraction.shape + (3,), dtype=np.uint8)
    raw[..., 0] = np.rint(np.asarray(fraction) * 128).astype(np.uint8)
    raw[..., 1:] = [2, 255]
    meta = {"dimensions_xyz": list(fraction.shape), "voxel_size_m": spacing,
            "origin_m": list(origin), "has_plane_equations": False}
    return raw, meta


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("level", [0., .5])
def test_mask_keeps_crossings_on_block_edges_and_exact_level_ties(step, level):
    field = np.full((43, 39, 35), level - 1, dtype=np.float32)
    # The feature crosses block boundaries at 8, 16 and 24; a plane on a
    # block boundary exactly touches the isovalue rather than straddling it.
    field[8:25, 8:24, 7:22] = level + 1
    field[8, 8:24, 7:22] = level
    field[24, 8:24, 7:22] = level
    field[17:20, 12:15, 10:13] = level - .25
    mask, report = _active_block_mask(field, level, step_size=step, block_size=8)
    reference = marching_cubes(field, level, step_size=step, allow_degenerate=False)
    masked = marching_cubes(field, level, step_size=step, allow_degenerate=False, mask=mask)
    np.testing.assert_array_equal(masked[0], reference[0])
    np.testing.assert_array_equal(masked[1], reference[1])
    assert report["mask_coverage_fraction"] < .8
    assert report["mask_bytes"] == field.size
    assert report["preprocessing_seconds"] >= 0
    assert not report["geometry_samples_modified"]


@pytest.mark.parametrize("surface", ["fraction", "vof", "plane"])
@pytest.mark.parametrize("crop", [False, True])
def test_material_modes_padding_crop_origin_and_threshold_ties(surface, crop):
    # Axis-aligned mixed surface includes an exactly half-filled layer.
    c = np.zeros((73, 70, 68), dtype=np.float32)
    c[14:35, 11:57, 10:55] = 1
    c[35, 11:57, 10:55] = .5
    c[36, 11:57, 10:55] = .25
    raw, meta = encoded(c)
    kwargs = {"surface": surface}
    if surface == "plane":
        planes = np.zeros(c.shape + (4,), dtype=np.float32)
        planes[..., 0] = 1
        # Preserve the supplied-plane field without requiring its offset to
        # coincide with the occupancy-derived offset.  Exact-level masked MC
        # ties are covered separately, including an existing audit rejection.
        planes[..., 3] = 35.625 - np.arange(c.shape[0])[:, None, None]
        kwargs["planes"] = planes
        meta["has_plane_equations"] = True
    if crop:
        kwargs["crop_indices"] = ((5, 66), (7, 65), (3, 63))
    reference = mesh_material(raw, meta, 2, **kwargs)
    masked = mesh_material(raw, meta, 2, active_blocks=True, **kwargs)
    np.testing.assert_array_equal(masked[0], reference[0])
    np.testing.assert_array_equal(masked[1], reference[1])
    np.testing.assert_array_equal(masked[2]["translation_m"], reference[2]["translation_m"])
    for key in ("boundary_edges", "nonmanifold_edges", "inconsistent_winding_edges",
                "degenerate_faces"):
        assert masked[2][key] == reference[2][key] == 0
    assert masked[2]["watertight_edges"]
    assert masked[2]["active_blocks"]["global_extraction"]
    assert reference[2]["active_blocks"] == {"enabled": False}


@pytest.mark.parametrize("step", [1, 2, 3])
def test_all_solid_outer_caps_and_stride_padding_unchanged(step):
    raw, meta = encoded(np.ones((68, 70, 72)), spacing=[10e-9, 20e-9, 30e-9])
    reference = mesh_material(raw, meta, 2, step_size=step)
    masked = mesh_material(raw, meta, 2, active_blocks=True, step_size=step)
    np.testing.assert_array_equal(masked[0], reference[0])
    np.testing.assert_array_equal(masked[1], reference[1])
    assert masked[2]["active_blocks"]["mask_coverage_fraction"] < 1
    assert masked[2]["watertight_edges"]


def test_empty_field_skips_mask_and_rejects_nonboolean_flag():
    raw, meta = encoded(np.zeros((4, 5, 6)))
    vertices, faces, report = mesh_material(raw, meta, 2, active_blocks=True)
    assert len(vertices) == len(faces) == 0
    assert report["active_blocks"]["skipped_reason"] == "empty material field"
    with pytest.raises(MeshingError, match="boolean"):
        mesh_material(raw, meta, 2, active_blocks="auto")


def test_uniform_threshold_block_has_no_surface_or_mask():
    mask, report = _active_block_mask(np.zeros((9, 10, 11), np.float32), 0., block_size=4)
    assert not np.any(mask)
    assert report["active_blocks"] == 0


def test_existing_plane_tie_nonmanifold_is_still_rejected():
    c = np.zeros((39, 37, 35), dtype=np.float32)
    c[4:21, 4:30, 4:29] = 1
    c[21, 4:30, 4:29] = .5
    c[22, 4:30, 4:29] = .25
    raw, meta = encoded(c)
    planes = np.zeros(c.shape + (4,), dtype=np.float32)
    planes[..., 0] = 1
    planes[..., 3] = 21.5 - np.arange(c.shape[0])[:, None, None]
    # This corner/tie pattern already fails the unmasked topology audit.
    # Optimization must preserve that rejection, rather than silently patch
    # the field or relax the validation in order to produce an output.
    for active in (False, True):
        with pytest.raises(MeshingError, match="nonmanifold_edges=2"):
            mesh_material(raw, meta, 2, surface="plane", planes=planes,
                          active_blocks=active)


@pytest.mark.parametrize("kwargs", [{"block_size": 0}, {"block_size": True},
                                    {"step_size": 0}, {"step_size": 4}])
def test_invalid_mask_configuration(kwargs):
    with pytest.raises(MeshingError):
        _active_block_mask(np.zeros((4, 4, 4), np.float32), 0., **kwargs)
