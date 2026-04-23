# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for adjoints of in-place composite-component writes.

Covers the shapes the v2 lowering currently handles — array-rooted writes
where the access chain from ``arr[i]`` lands on a composite slot, single
assignment (``=``). Augmented assignments, nested struct chains, and
local-struct composite-field writes are tracked as follow-ups.

See ``asv/benchmarks/codegen/composite_component_*.py`` for the
performance contract that these tests complement.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.struct
class Scalar:
    a: wp.float32
    b: wp.float32


@wp.struct
class StateStruct:
    position: wp.vec3
    velocity: wp.vec3


@wp.kernel
def _seed_scalar_grad(g: wp.array(dtype=Scalar), a_val: wp.float32, b_val: wp.float32):
    i = wp.tid()
    g[i] = Scalar(a_val, b_val)


@wp.kernel
def _seed_state_grad(g: wp.array(dtype=StateStruct), pos: wp.vec3, vel: wp.vec3):
    i = wp.tid()
    s = StateStruct()
    s.position = pos
    s.velocity = vel
    g[i] = s


# --------- kernels under test ---------


@wp.kernel
def _k_gh583(dst: wp.array(dtype=wp.vec3), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i].y = src[i]


@wp.kernel
def _k_gh248(y: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    y[tid].x = x[tid] * 2.0
    y[tid].y = x[tid] * 3.0
    y[tid].z = x[tid] * 4.0


@wp.kernel
def _k_gh1174(y: wp.array(dtype=Scalar), x: wp.array(dtype=wp.float32)):
    i = wp.tid()
    y[i].a = x[i]


@wp.kernel
def _k_vec2_component(y: wp.array(dtype=wp.vec2), x: wp.array(dtype=wp.float32)):
    i = wp.tid()
    y[i].y = x[i]


@wp.kernel
def _k_vec4_component(y: wp.array(dtype=wp.vec4), x: wp.array(dtype=wp.float32)):
    i = wp.tid()
    y[i].z = x[i]


@wp.kernel
def _k_quat_component(y: wp.array(dtype=wp.quatf), x: wp.array(dtype=wp.float32)):
    i = wp.tid()
    y[i].x = x[i]


@wp.kernel
def _k_mat22_element(y: wp.array(dtype=wp.mat22), x: wp.array(dtype=wp.float32)):
    i = wp.tid()
    y[i][0, 1] = x[i]


@wp.kernel
def _k_mat33_element(y: wp.array(dtype=wp.mat33), x: wp.array(dtype=wp.float32)):
    i = wp.tid()
    y[i][1, 2] = x[i]


@wp.kernel
def _k_transform_p(y: wp.array(dtype=wp.transformf), p: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    y[i].p = p[i]


@wp.kernel
def _k_transform_q(y: wp.array(dtype=wp.transformf), q: wp.array(dtype=wp.quatf)):
    i = wp.tid()
    y[i].q = q[i]


@wp.kernel
def _k_struct_vec_field(y: wp.array(dtype=StateStruct), p: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    y[i].position = p[i]


@wp.kernel
def _k_array2d(y: wp.array2d(dtype=wp.vec3), x: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    y[i, j].y = x[i, j]


@wp.kernel
def _k_array3d(y: wp.array3d(dtype=wp.vec3), x: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    y[i, j, k].z = x[i, j, k]


# --------- tests ---------


class TestCompositeComponentAdjoint(unittest.TestCase):
    def test_gh583_array_vec_component(self):
        n = 4
        src = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        dst = wp.zeros(n, dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_gh583, n, inputs=[dst, src])

        dst.grad = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        np.testing.assert_allclose(src.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_gh248_sequential_component_writes(self):
        n = 3
        x = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_gh248, n, inputs=[y, x])

        y.grad = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.full(n, 9.0, dtype=np.float32))

    def test_gh1174_array_struct_field(self):
        n = 3
        x = wp.array(np.ones(n, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=Scalar, requires_grad=True)

        wp.launch(_seed_scalar_grad, n, inputs=[y.grad, 1.0, 0.0])

        tape = wp.Tape()
        with tape:
            wp.launch(_k_gh1174, n, inputs=[y, x])
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_vec2_component_write(self):
        n = 2
        x = wp.array(np.full(n, 2.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.vec2, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_vec2_component, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 2), dtype=np.float32), dtype=wp.vec2)
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_vec4_component_write(self):
        n = 2
        x = wp.array(np.full(n, 1.5, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.vec4, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_vec4_component, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 4), dtype=np.float32), dtype=wp.vec4)
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_quaternion_component_write(self):
        n = 2
        x = wp.array(np.full(n, 0.5, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.quatf, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_quat_component, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 4), dtype=np.float32), dtype=wp.quatf)
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_mat22_element_write(self):
        n = 2
        x = wp.array(np.full(n, 3.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.mat22, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_mat22_element, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 2, 2), dtype=np.float32), dtype=wp.mat22)
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_mat33_element_write(self):
        n = 2
        x = wp.array(np.full(n, 7.0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros(n, dtype=wp.mat33, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_mat33_element, n, inputs=[y, x])
        y.grad = wp.array(np.ones((n, 3, 3), dtype=np.float32), dtype=wp.mat33)
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.ones(n, dtype=np.float32))

    def test_transform_translation_write(self):
        n = 2
        p = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, requires_grad=True)
        y = wp.zeros(n, dtype=wp.transformf, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_transform_p, n, inputs=[y, p])
        y.grad = wp.array(np.ones((n, 7), dtype=np.float32), dtype=wp.transformf)
        tape.backward()

        np.testing.assert_allclose(p.grad.numpy(), np.ones((n, 3), dtype=np.float32))

    def test_transform_rotation_write(self):
        n = 2
        q = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32), dtype=wp.quatf, requires_grad=True)
        y = wp.zeros(n, dtype=wp.transformf, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_transform_q, n, inputs=[y, q])
        y.grad = wp.array(np.ones((n, 7), dtype=np.float32), dtype=wp.transformf)
        tape.backward()

        np.testing.assert_allclose(q.grad.numpy(), np.ones((n, 4), dtype=np.float32))

    def test_struct_vec_field_write(self):
        n = 2
        p = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, requires_grad=True)
        y = wp.zeros(n, dtype=StateStruct, requires_grad=True)

        wp.launch(_seed_state_grad, n, inputs=[y.grad, wp.vec3(1.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0)])

        tape = wp.Tape()
        with tape:
            wp.launch(_k_struct_vec_field, n, inputs=[y, p])
        tape.backward()

        np.testing.assert_allclose(p.grad.numpy(), np.ones((n, 3), dtype=np.float32))

    def test_array2d_composite_component(self):
        rows, cols = 2, 3
        x = wp.array(np.ones((rows, cols), dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros((rows, cols), dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_array2d, (rows, cols), inputs=[y, x])
        y.grad = wp.array(np.ones((rows, cols, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.ones((rows, cols), dtype=np.float32))

    def test_array3d_composite_component(self):
        a, b, c = 2, 2, 2
        x = wp.array(np.ones((a, b, c), dtype=np.float32), dtype=wp.float32, requires_grad=True)
        y = wp.zeros((a, b, c), dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(_k_array3d, (a, b, c), inputs=[y, x])
        y.grad = wp.array(np.ones((a, b, c, 3), dtype=np.float32), dtype=wp.vec3)
        tape.backward()

        np.testing.assert_allclose(x.grad.numpy(), np.ones((a, b, c), dtype=np.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
