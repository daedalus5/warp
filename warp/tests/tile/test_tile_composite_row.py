# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import (
    add_function_test,
    assert_np_equal,
    get_cuda_test_devices,
)

# Number of threads per tile block.  Each thread reads one mat33 element from
# the tile and writes one vec3 row to the output array.
BLOCK_DIM = 8

# ── 1D tile of mat33: extract row by index ───────────────────────────────────
# Load a 1D tile-of-mat33 of size BLOCK_DIM from an array.  Each thread reads
# row 1 (= [4, 5, 6]) from its own tile slot and stores it into `out`.
#
# Kernel signature: `_tile, i = wp.tid()` where `_tile` is the outer tile-block
# coordinate (always 0 here) and `i` is the per-thread index 0..BLOCK_DIM-1.


@wp.kernel
def _k_tile_mat33_row_read_1d(
    inp: wp.array(dtype=wp.mat33),
    out: wp.array(dtype=wp.vec3),
):
    _tile, i = wp.tid()
    t = wp.tile_load(inp, shape=(BLOCK_DIM,), offset=(BLOCK_DIM * _tile,))
    out[i] = t[i][1]


def test_tile_mat33_row_read_1d(test, device):
    n = BLOCK_DIM
    # every matrix = [[1,2,3],[4,5,6],[7,8,9]]
    data = np.tile(np.arange(1.0, 10.0, dtype=np.float32).reshape(3, 3), (n, 1, 1))
    inp = wp.array(data, dtype=wp.mat33, device=device)
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch_tiled(_k_tile_mat33_row_read_1d, dim=[1], inputs=[inp, out], block_dim=BLOCK_DIM, device=device)
    wp.synchronize_device()
    expected = np.tile([4.0, 5.0, 6.0], (n, 1))
    assert_np_equal(out.numpy(), expected)


# ── 2D tile of mat33: extract row by index ───────────────────────────────────
# A 2x2 tile-of-mat33 accessed from a 2D kernel (tile block indices i, j).
# Each of the 4 threads in the BLOCK_DIM reads its (i%2, j%2) position and
# extracts row 2 = [7, 8, 9].
#
# Grid layout with launch_tiled:  dim=[1, 1] + block_dim=4 → [1, 1, 4]
# wp.tid() → (tile_i, tile_j, thread_idx).  We unpack as:
#   _bi, _bj, idx = wp.tid()
# and derive 2D row/col from idx.
BLOCK_DIM_2D = 4


@wp.kernel
def _k_tile_mat33_row_read_2d(
    inp: wp.array2d(dtype=wp.mat33),
    out: wp.array2d(dtype=wp.vec3),
):
    _bi, _bj, idx = wp.tid()
    row = idx // 2
    col = idx % 2
    t = wp.tile_load(inp, shape=(2, 2))
    out[row, col] = t[row, col][2]  # row 2 = [7, 8, 9]


def test_tile_mat33_row_read_2d(test, device):
    rows, cols = 2, 2
    data = np.tile(np.arange(1.0, 10.0, dtype=np.float32).reshape(3, 3), (rows, cols, 1, 1))
    inp = wp.array(data, dtype=wp.mat33, device=device)
    out = wp.zeros((rows, cols), dtype=wp.vec3, device=device)
    wp.launch_tiled(_k_tile_mat33_row_read_2d, dim=[1, 1], inputs=[inp, out], block_dim=BLOCK_DIM_2D, device=device)
    wp.synchronize_device()
    expected = np.tile([7.0, 8.0, 9.0], (rows, cols, 1))
    assert_np_equal(out.numpy(), expected)


# ── 3D tile of mat33: extract row by index ───────────────────────────────────
# A (2,2,2) tile-of-mat33. Each of the 8 threads decodes its flat index into
# (i,j,k), loads row 1 = [4,5,6] from its tile slot, and writes to out[i,j,k].
D0_3D = 2
D1_3D = 2
D2_3D = 2
BLOCK_DIM_3D = D0_3D * D1_3D * D2_3D


@wp.kernel
def _k_tile_mat33_row_read_3d(
    inp: wp.array3d(dtype=wp.mat33),
    out: wp.array3d(dtype=wp.vec3),
):
    _tile, idx = wp.tid()
    i = idx // (D1_3D * D2_3D)
    j = (idx // D2_3D) % D1_3D
    k = idx % D2_3D
    t = wp.tile_load(inp, shape=(D0_3D, D1_3D, D2_3D))
    out[i, j, k] = t[i, j, k][1]  # row 1 = [4, 5, 6]


def test_tile_mat33_row_read_3d(test, device):
    data = np.tile(
        np.arange(1.0, 10.0, dtype=np.float32).reshape(3, 3),
        (D0_3D, D1_3D, D2_3D, 1, 1),
    )
    inp = wp.array(data, dtype=wp.mat33, device=device)
    out = wp.zeros((D0_3D, D1_3D, D2_3D), dtype=wp.vec3, device=device)
    wp.launch_tiled(
        _k_tile_mat33_row_read_3d,
        dim=[1],
        inputs=[inp, out],
        block_dim=BLOCK_DIM_3D,
        device=device,
    )
    wp.synchronize_device()
    expected = np.tile([4.0, 5.0, 6.0], (D0_3D, D1_3D, D2_3D, 1))
    assert_np_equal(out.numpy(), expected)


# ── 4D tile of mat22: extract row by index ───────────────────────────────────
# A (2,2,2,2) tile-of-mat22. Each of the 16 threads decodes its flat index into
# (i,j,k,l), loads row 0 = [1,2] from its tile slot, and writes to out[i,j,k,l].
D0_4D = 2
D1_4D = 2
D2_4D = 2
D3_4D = 2
BLOCK_DIM_4D = D0_4D * D1_4D * D2_4D * D3_4D


@wp.kernel
def _k_tile_mat22_row_read_4d(
    inp: wp.array4d(dtype=wp.mat22),
    out: wp.array4d(dtype=wp.vec2),
):
    _tile, idx = wp.tid()
    i = idx // (D1_4D * D2_4D * D3_4D)
    j = (idx // (D2_4D * D3_4D)) % D1_4D
    k = (idx // D3_4D) % D2_4D
    l = idx % D3_4D
    t = wp.tile_load(inp, shape=(D0_4D, D1_4D, D2_4D, D3_4D))
    out[i, j, k, l] = t[i, j, k, l][0]  # row 0 = [1, 2]


def test_tile_mat22_row_read_4d(test, device):
    data = np.tile(
        np.arange(1.0, 5.0, dtype=np.float32).reshape(2, 2),
        (D0_4D, D1_4D, D2_4D, D3_4D, 1, 1),
    )
    inp = wp.array(data, dtype=wp.mat22, device=device)
    out = wp.zeros((D0_4D, D1_4D, D2_4D, D3_4D), dtype=wp.vec2, device=device)
    wp.launch_tiled(
        _k_tile_mat22_row_read_4d,
        dim=[1],
        inputs=[inp, out],
        block_dim=BLOCK_DIM_4D,
        device=device,
    )
    wp.synchronize_device()
    expected = np.tile([1.0, 2.0], (D0_4D, D1_4D, D2_4D, D3_4D, 1))
    assert_np_equal(out.numpy(), expected)


# ── Adjoint test: backward through 1D tile-of-mat row read ──────────────────
# Forward: load a BLOCK_DIM-element tile of mat22; each thread extracts row 0
#          from its slot, writing [m[0,0], m[0,1]] to out[i].
# Backward: inject gradient [1, 1] at every output position; the adjoint of
#           the row-extract should write [1, 1] back into row 0 of the
#           corresponding input-matrix gradient, leaving row 1 at zero.


@wp.kernel
def _k_tile_mat22_row_read_adj(
    inp: wp.array(dtype=wp.mat22),
    out: wp.array(dtype=wp.vec2),
):
    _tile, i = wp.tid()
    t = wp.tile_load(inp, shape=(BLOCK_DIM,), offset=(BLOCK_DIM * _tile,))
    out[i] = t[i][0]


def test_tile_mat22_row_read_backward(test, device):
    n = BLOCK_DIM
    init = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)
    inp = wp.array(init, dtype=wp.mat22, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.vec2, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch_tiled(_k_tile_mat22_row_read_adj, dim=[1], inputs=[inp, out], block_dim=n, device=device)

    out.grad = wp.array(np.tile([1.0, 1.0], (n, 1)), dtype=wp.vec2, device=device)
    tape.backward()

    expected = np.zeros_like(init)
    expected[:, 0, :] = 1.0
    assert_np_equal(inp.grad.numpy(), expected)


# ── 1D tile of mat33: write row by index ─────────────────────────────────────
# Allocate a zero tile of mat33 (size BLOCK_DIM).  Each thread writes row 1 of
# its slot to [10, 20, 30], then stores the tile element to the output array.


@wp.kernel
def _k_tile_mat33_row_write_1d(
    out: wp.array(dtype=wp.mat33),
):
    _tile, i = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat33, shape=(BLOCK_DIM,))
    t[i][1] = wp.vec3(10.0, 20.0, 30.0)
    out[i] = t[i]


def test_tile_mat33_row_write_1d(test, device):
    out = wp.zeros(BLOCK_DIM, dtype=wp.mat33, device=device)
    wp.launch_tiled(_k_tile_mat33_row_write_1d, dim=[1], inputs=[out], block_dim=BLOCK_DIM, device=device)
    wp.synchronize_device()
    expected = np.zeros((BLOCK_DIM, 3, 3), dtype=np.float32)
    expected[:, 1, :] = [10.0, 20.0, 30.0]
    assert_np_equal(out.numpy(), expected)


# ── 2D tile of mat33: write row by index ─────────────────────────────────────
# A 2x2 tile-of-mat33.  Each thread writes row 0 of its slot to [1, 2, 3].


@wp.kernel
def _k_tile_mat33_row_write_2d(
    out: wp.array2d(dtype=wp.mat33),
):
    _bi, _bj, idx = wp.tid()
    row = idx // 2
    col = idx % 2
    t = wp.tile_zeros(dtype=wp.mat33, shape=(2, 2))
    t[row, col][0] = wp.vec3(1.0, 2.0, 3.0)
    out[row, col] = t[row, col]


def test_tile_mat33_row_write_2d(test, device):
    rows, cols = 2, 2
    out = wp.zeros((rows, cols), dtype=wp.mat33, device=device)
    wp.launch_tiled(_k_tile_mat33_row_write_2d, dim=[1, 1], inputs=[out], block_dim=BLOCK_DIM_2D, device=device)
    wp.synchronize_device()
    expected = np.zeros((rows, cols, 3, 3), dtype=np.float32)
    expected[:, :, 0, :] = [1.0, 2.0, 3.0]
    assert_np_equal(out.numpy(), expected)


# ── 3D tile of mat33: write row by index ─────────────────────────────────────
# A (2,2,2) tile-of-mat33.  Each thread writes row 2 of its slot to [7, 8, 9].


@wp.kernel
def _k_tile_mat33_row_write_3d(
    out: wp.array3d(dtype=wp.mat33),
):
    _tile, idx = wp.tid()
    i = idx // (D1_3D * D2_3D)
    j = (idx // D2_3D) % D1_3D
    k = idx % D2_3D
    t = wp.tile_zeros(dtype=wp.mat33, shape=(D0_3D, D1_3D, D2_3D))
    t[i, j, k][2] = wp.vec3(7.0, 8.0, 9.0)
    out[i, j, k] = t[i, j, k]


def test_tile_mat33_row_write_3d(test, device):
    out = wp.zeros((D0_3D, D1_3D, D2_3D), dtype=wp.mat33, device=device)
    wp.launch_tiled(
        _k_tile_mat33_row_write_3d,
        dim=[1],
        inputs=[out],
        block_dim=BLOCK_DIM_3D,
        device=device,
    )
    wp.synchronize_device()
    expected = np.zeros((D0_3D, D1_3D, D2_3D, 3, 3), dtype=np.float32)
    expected[:, :, :, 2, :] = [7.0, 8.0, 9.0]
    assert_np_equal(out.numpy(), expected)


# ── 4D tile of mat22: write row by index ─────────────────────────────────────
# A (2,2,2,2) tile-of-mat22.  Each thread writes row 1 of its slot to [3, 4].


@wp.kernel
def _k_tile_mat22_row_write_4d(
    out: wp.array4d(dtype=wp.mat22),
):
    _tile, idx = wp.tid()
    i = idx // (D1_4D * D2_4D * D3_4D)
    j = (idx // (D2_4D * D3_4D)) % D1_4D
    k = (idx // D3_4D) % D2_4D
    l = idx % D3_4D
    t = wp.tile_zeros(dtype=wp.mat22, shape=(D0_4D, D1_4D, D2_4D, D3_4D))
    t[i, j, k, l][1] = wp.vec2(3.0, 4.0)
    out[i, j, k, l] = t[i, j, k, l]


def test_tile_mat22_row_write_4d(test, device):
    out = wp.zeros((D0_4D, D1_4D, D2_4D, D3_4D), dtype=wp.mat22, device=device)
    wp.launch_tiled(
        _k_tile_mat22_row_write_4d,
        dim=[1],
        inputs=[out],
        block_dim=BLOCK_DIM_4D,
        device=device,
    )
    wp.synchronize_device()
    expected = np.zeros((D0_4D, D1_4D, D2_4D, D3_4D, 2, 2), dtype=np.float32)
    expected[:, :, :, :, 1, :] = [3.0, 4.0]
    assert_np_equal(out.numpy(), expected)


# ── Adjoint test: backward through 1D tile-of-mat row write ─────────────────
# Forward: load a BLOCK_DIM-element tile of mat22 from a zero input; each
#          thread writes row 1 from the src array into its tile slot, then
#          stores back.
# Backward: inject gradient [[0,0],[1,1]] at every output position; the
#           adjoint of the row-write should propagate the row-1 gradient back
#           to the src vector.


@wp.kernel
def _k_tile_mat22_row_write_adj(
    src_rows: wp.array(dtype=wp.vec2),
    out: wp.array(dtype=wp.mat22),
):
    _tile, i = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat22, shape=(BLOCK_DIM,))
    t[i][1] = src_rows[i]
    out[i] = t[i]


def test_tile_mat22_row_write_backward(test, device):
    n = BLOCK_DIM
    src_data = np.ones((n, 2), dtype=np.float32)
    src_rows = wp.array(src_data, dtype=wp.vec2, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch_tiled(_k_tile_mat22_row_write_adj, dim=[1], inputs=[src_rows, out], block_dim=n, device=device)

    # Seed grad: [[0,0],[g0,g1]] at every output element
    grad_out = np.zeros((n, 2, 2), dtype=np.float32)
    grad_out[:, 1, :] = [2.0, 3.0]
    out.grad = wp.array(grad_out, dtype=wp.mat22, device=device)
    tape.backward()

    # Gradient of row 1 of each matrix should flow back to src_rows
    expected = np.tile([2.0, 3.0], (n, 1))
    assert_np_equal(src_rows.grad.numpy(), expected)


# ── 1D tile of mat33: augmented-assign row (+=) ──────────────────────────────
# Load a BLOCK_DIM-element tile of mat33 where every element is all-ones.
# Each thread adds vec3(1, 2, 3) to row 1 of its slot, then stores back.


@wp.kernel
def _k_tile_mat33_row_iadd_1d(
    init: wp.array(dtype=wp.mat33),
    out: wp.array(dtype=wp.mat33),
):
    _tile, i = wp.tid()
    t = wp.tile_load(init, shape=(BLOCK_DIM,), offset=(BLOCK_DIM * _tile,))
    t[i][1] += wp.vec3(1.0, 2.0, 3.0)
    out[i] = t[i]


def test_tile_mat33_row_iadd_1d(test, device):
    n = BLOCK_DIM
    init = np.ones((n, 3, 3), dtype=np.float32)
    inp = wp.array(init, dtype=wp.mat33, device=device)
    out = wp.zeros(n, dtype=wp.mat33, device=device)
    wp.launch_tiled(_k_tile_mat33_row_iadd_1d, dim=[1], inputs=[inp, out], block_dim=n, device=device)
    wp.synchronize_device()
    expected = init.copy()
    expected[:, 1, :] += [1.0, 2.0, 3.0]
    assert_np_equal(out.numpy(), expected)


# ── 1D tile of mat33: augmented-assign row (-=) ──────────────────────────────
# Each thread subtracts vec3(1, 2, 3) from row 0 of its slot.


@wp.kernel
def _k_tile_mat33_row_isub_1d(
    init: wp.array(dtype=wp.mat33),
    out: wp.array(dtype=wp.mat33),
):
    _tile, i = wp.tid()
    t = wp.tile_load(init, shape=(BLOCK_DIM,), offset=(BLOCK_DIM * _tile,))
    t[i][0] -= wp.vec3(1.0, 2.0, 3.0)
    out[i] = t[i]


def test_tile_mat33_row_isub_1d(test, device):
    n = BLOCK_DIM
    init = np.ones((n, 3, 3), dtype=np.float32)
    inp = wp.array(init, dtype=wp.mat33, device=device)
    out = wp.zeros(n, dtype=wp.mat33, device=device)
    wp.launch_tiled(_k_tile_mat33_row_isub_1d, dim=[1], inputs=[inp, out], block_dim=n, device=device)
    wp.synchronize_device()
    expected = init.copy()
    expected[:, 0, :] -= [1.0, 2.0, 3.0]
    assert_np_equal(out.numpy(), expected)


# ── 2D tile of mat33: augmented-assign row (+=) ───────────────────────────────


@wp.kernel
def _k_tile_mat33_row_iadd_2d(
    init: wp.array2d(dtype=wp.mat33),
    out: wp.array2d(dtype=wp.mat33),
):
    _bi, _bj, idx = wp.tid()
    row = idx // 2
    col = idx % 2
    t = wp.tile_load(init, shape=(2, 2))
    t[row, col][2] += wp.vec3(10.0, 20.0, 30.0)
    out[row, col] = t[row, col]


def test_tile_mat33_row_iadd_2d(test, device):
    rows, cols = 2, 2
    init = np.ones((rows, cols, 3, 3), dtype=np.float32)
    inp = wp.array(init, dtype=wp.mat33, device=device)
    out = wp.zeros((rows, cols), dtype=wp.mat33, device=device)
    wp.launch_tiled(_k_tile_mat33_row_iadd_2d, dim=[1, 1], inputs=[inp, out], block_dim=BLOCK_DIM_2D, device=device)
    wp.synchronize_device()
    expected = init.copy()
    expected[:, :, 2, :] += [10.0, 20.0, 30.0]
    assert_np_equal(out.numpy(), expected)


# ── 3D tile of mat33: augmented-assign row (+=) ───────────────────────────────


@wp.kernel
def _k_tile_mat33_row_iadd_3d(
    init: wp.array3d(dtype=wp.mat33),
    out: wp.array3d(dtype=wp.mat33),
):
    _tile, idx = wp.tid()
    i = idx // (D1_3D * D2_3D)
    j = (idx // D2_3D) % D1_3D
    k = idx % D2_3D
    t = wp.tile_load(init, shape=(D0_3D, D1_3D, D2_3D))
    t[i, j, k][0] += wp.vec3(5.0, 6.0, 7.0)
    out[i, j, k] = t[i, j, k]


def test_tile_mat33_row_iadd_3d(test, device):
    init = np.ones((D0_3D, D1_3D, D2_3D, 3, 3), dtype=np.float32)
    inp = wp.array(init, dtype=wp.mat33, device=device)
    out = wp.zeros((D0_3D, D1_3D, D2_3D), dtype=wp.mat33, device=device)
    wp.launch_tiled(
        _k_tile_mat33_row_iadd_3d,
        dim=[1],
        inputs=[inp, out],
        block_dim=BLOCK_DIM_3D,
        device=device,
    )
    wp.synchronize_device()
    expected = init.copy()
    expected[:, :, :, 0, :] += [5.0, 6.0, 7.0]
    assert_np_equal(out.numpy(), expected)


# ── Adjoint test: backward through 1D tile-of-mat row iadd (+=) ─────────────
# Forward: load a BLOCK_DIM-element tile of mat22; each thread does
#          t[i][1] += src_rows[i] and stores t[i] to out.
# Backward: inject gradient [[0,0],[g0,g1]] at out; the adjoint of iadd
#           accumulates the row-1 gradient into adj_src_rows (not zeroed,
#           unlike assign: adj_src += adj_out row).


@wp.kernel
def _k_tile_mat22_row_iadd_adj(
    src_rows: wp.array(dtype=wp.vec2),
    out: wp.array(dtype=wp.mat22),
):
    _tile, i = wp.tid()
    t = wp.tile_zeros(dtype=wp.mat22, shape=(BLOCK_DIM,))
    t[i][1] += src_rows[i]
    out[i] = t[i]


def test_tile_mat22_row_iadd_backward(test, device):
    n = BLOCK_DIM
    src_data = np.ones((n, 2), dtype=np.float32)
    src_rows = wp.array(src_data, dtype=wp.vec2, requires_grad=True, device=device)
    out = wp.zeros(n, dtype=wp.mat22, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch_tiled(_k_tile_mat22_row_iadd_adj, dim=[1], inputs=[src_rows, out], block_dim=n, device=device)

    # Seed gradient: only row 1 of each output matrix has non-zero grad
    grad_out = np.zeros((n, 2, 2), dtype=np.float32)
    grad_out[:, 1, :] = [4.0, 5.0]
    out.grad = wp.array(grad_out, dtype=wp.mat22, device=device)
    tape.backward()

    # adj_src_rows should receive the row-1 gradient from the adjoint tile
    expected = np.tile([4.0, 5.0], (n, 1))
    assert_np_equal(src_rows.grad.numpy(), expected)


devices = get_cuda_test_devices()


class TestTileCompositeRow(unittest.TestCase):
    pass


add_function_test(TestTileCompositeRow, "test_tile_mat33_row_read_1d", test_tile_mat33_row_read_1d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_read_2d", test_tile_mat33_row_read_2d, devices=devices)
add_function_test(
    TestTileCompositeRow, "test_tile_mat22_row_read_backward", test_tile_mat22_row_read_backward, devices=devices
)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_read_3d", test_tile_mat33_row_read_3d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat22_row_read_4d", test_tile_mat22_row_read_4d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_write_1d", test_tile_mat33_row_write_1d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_write_2d", test_tile_mat33_row_write_2d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_write_3d", test_tile_mat33_row_write_3d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat22_row_write_4d", test_tile_mat22_row_write_4d, devices=devices)
add_function_test(
    TestTileCompositeRow, "test_tile_mat22_row_write_backward", test_tile_mat22_row_write_backward, devices=devices
)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_iadd_1d", test_tile_mat33_row_iadd_1d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_isub_1d", test_tile_mat33_row_isub_1d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_iadd_2d", test_tile_mat33_row_iadd_2d, devices=devices)
add_function_test(TestTileCompositeRow, "test_tile_mat33_row_iadd_3d", test_tile_mat33_row_iadd_3d, devices=devices)
add_function_test(
    TestTileCompositeRow, "test_tile_mat22_row_iadd_backward", test_tile_mat22_row_iadd_backward, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
