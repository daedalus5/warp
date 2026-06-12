# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_conditional_if_else():
    a = 0.5
    b = 2.0

    if a > b:
        c = 1.0
    else:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_conditional_if_else_nested():
    a = 1.0
    b = 2.0

    if a > b:
        c = 3.0
        d = 4.0

        if c > d:
            e = 1.0
        else:
            e = -1.0

    else:
        c = 6.0
        d = 7.0

        if c > d:
            e = 2.0
        else:
            e = -2.0

    wp.expect_eq(e, -2.0)


@wp.kernel
def test_conditional_ifexp():
    a = 0.5
    b = 2.0

    c = 1.0 if a > b else -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_conditional_ifexp_nested():
    a = 1.0
    b = 2.0

    c = 3.0 if a > b else 6.0
    d = 4.0 if a > b else 7.0
    e = 1.0 if (a > b and c > d) else (-1.0 if a > b else (2.0 if c > d else -2.0))

    wp.expect_eq(e, -2.0)


@wp.kernel
def test_conditional_ifexp_constant():
    a = 1.0 if False else -1.0
    b = 2.0 if 123 else -2.0

    wp.expect_eq(a, -1.0)
    wp.expect_eq(b, 2.0)


@wp.kernel
def test_conditional_ifexp_constant_nested():
    a = 1.0 if False else (2.0 if True else 3.0)
    b = 4.0 if 0 else (5.0 if 0 else (6.0 if False else 7.0))
    c = 8.0 if False else (9.0 if False else (10.0 if 321 else 11.0))

    wp.expect_eq(a, 2.0)
    wp.expect_eq(b, 7.0)
    wp.expect_eq(c, 10.0)


@wp.kernel
def test_boolean_and():
    a = 1.0
    b = 2.0
    c = 1.0

    if a > 0.0 and b > 0.0:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_boolean_or():
    a = 1.0
    b = 2.0
    c = 1.0

    if a > 0.0 and b > 0.0:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_boolean_compound():
    a = 1.0
    b = 2.0
    c = 3.0

    d = 1.0

    if (a > 0.0 and b > 0.0) or c > a:
        d = -1.0

    wp.expect_eq(d, -1.0)


@wp.kernel
def test_boolean_literal():
    t = True
    f = False

    r = 1.0

    if t == (not f):
        r = -1.0

    wp.expect_eq(r, -1.0)


@wp.kernel
def test_int_logical_not():
    x = 0
    if not 123:
        x = 123

    wp.expect_eq(x, 0)


@wp.kernel
def test_int_conditional_assign_overload():
    if 123:
        x = 123

    if 234:
        x = 234

    wp.expect_eq(x, 234)


@wp.kernel
def test_bool_param_conditional(foo: bool):
    if foo:
        x = 123

    wp.expect_eq(x, 123)


@wp.kernel
def test_conditional_chain_basic():
    x = -1

    if 0 < x < 1:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_empty_range():
    x = -1
    y = 4

    if -2 <= x <= 10 <= y:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_faker():
    x = -1

    # Not actually a chained inequality
    if (-2 < x) < (1 > 0):
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_and():
    x = -1

    if (-2 < x < 0) and (-1 <= x <= -1):
        success = True
    else:
        success = False
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_eqs():
    x = wp.int32(10)
    y = 10
    z = -10

    if x == y != z:
        success = True
    else:
        success = False
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_mixed():
    x = 0

    if x < 10 == 1:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


def test_conditional_unequal_types(test: unittest.TestCase, device):
    # The bad kernel must be in a separate module, otherwise the current module would fail to load
    from warp.tests.aux_test_conditional_unequal_types_kernels import unequal_types_kernel  # noqa: PLC0415

    with test.assertRaises(TypeError):
        wp.launch(unequal_types_kernel, dim=(1,), inputs=[], device=device)

    # remove all references to the bad module so that subsequent calls to wp.force_load()
    # won't try to load it unless we explicitly re-import it again
    del wp._src.context.user_modules["warp.tests.aux_test_conditional_unequal_types_kernels"]
    del sys.modules["warp.tests.aux_test_conditional_unequal_types_kernels"]


@wp.kernel
def test_ifexp_with_array_access_kernel(
    idx: wp.int32,
    transforms: wp.array(dtype=wp.transform),
    result: wp.array(dtype=wp.vec3),
):
    # Conditional expression with array element access in else branch
    # When idx < 0, should use transform_identity() and NOT access transforms[idx]
    # This is the exact pattern that caused the segfault bug.
    t = wp.transform_identity() if idx < 0 else transforms[idx]
    result[0] = wp.transform_get_translation(t)


def test_ifexp_with_array_access(test: unittest.TestCase, device):
    transforms = wp.array((wp.transform(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),), dtype=wp.transform, device=device)
    result = wp.zeros(1, dtype=wp.vec3, device=device)

    wp.launch(
        test_ifexp_with_array_access_kernel,
        dim=1,
        inputs=(-1, transforms),
        outputs=(result,),
        device=device,
    )
    test.assertEqual(result.numpy()[0].tolist(), [0.0, 0.0, 0.0])

    wp.launch(
        test_ifexp_with_array_access_kernel,
        dim=1,
        inputs=(0, transforms),
        outputs=(result,),
        device=device,
    )
    test.assertEqual(result.numpy()[0].tolist(), [1.0, 2.0, 3.0])


@wp.kernel
def test_short_circuit_and_kernel(
    arr: wp.array(dtype=int),
    result: wp.array(dtype=int),
):
    tid = wp.tid()
    # arr[tid] must not be evaluated when arr is None (GH-1329)
    if arr and tid >= 0 and arr[tid] == 0:
        result[tid] = -1
        return
    result[tid] = 1


@wp.kernel
def test_short_circuit_or_kernel(
    arr: wp.array(dtype=int),
    result: wp.array(dtype=int),
):
    tid = wp.tid()
    # Second operand must not be evaluated when first is true
    if not arr or tid < 0 or arr[tid] == 0:
        result[tid] = -1
        return
    result[tid] = 1


def test_short_circuit_and(test: unittest.TestCase, device):
    """Chained `and` must short-circuit so null array is never dereferenced."""
    result = wp.zeros(3, dtype=int, device=device)
    # None array — should short-circuit, never access arr[tid]
    wp.launch(test_short_circuit_and_kernel, dim=3, inputs=[None, result], device=device)
    test.assertEqual(result.numpy().tolist(), [1, 1, 1])

    # Real array — should evaluate fully
    arr = wp.array([0, 1, 0], dtype=int, device=device)
    wp.launch(test_short_circuit_and_kernel, dim=3, inputs=[arr, result], device=device)
    test.assertEqual(result.numpy().tolist(), [-1, 1, -1])


def test_short_circuit_or(test: unittest.TestCase, device):
    """Chained `or` must short-circuit so null array is never dereferenced."""
    result = wp.zeros(3, dtype=int, device=device)
    # None array — `not arr` is true, should short-circuit
    wp.launch(test_short_circuit_or_kernel, dim=3, inputs=[None, result], device=device)
    test.assertEqual(result.numpy().tolist(), [-1, -1, -1])

    # Real array with non-zero values — all conditions false, result = 1
    arr = wp.array([5, 6, 7], dtype=int, device=device)
    wp.launch(test_short_circuit_or_kernel, dim=3, inputs=[arr, result], device=device)
    test.assertEqual(result.numpy().tolist(), [1, 1, 1])


@wp.kernel
def test_short_circuit_and_grad_kernel(
    x: wp.array(dtype=float),
    flag: wp.array(dtype=int),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    # flag[tid] != 0 and tid < 2: only threads 0,1 with flag set take the branch.
    # The backward pass must replay the same short-circuit guards so that
    # gradients flow only through the operands that were actually evaluated.
    if flag[tid] != 0 and tid < 2:
        out[tid] = x[tid] * 3.0
    else:
        out[tid] = x[tid] * 1.0


@wp.kernel
def test_short_circuit_or_grad_kernel(
    x: wp.array(dtype=float),
    flag: wp.array(dtype=int),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    # flag[tid] == 0 or tid >= 2: threads where flag is zero OR tid >= 2.
    if flag[tid] == 0 or tid >= 2:
        out[tid] = x[tid] * 1.0
    else:
        out[tid] = x[tid] * 3.0


def test_short_circuit_and_grad(test: unittest.TestCase, device):
    """Backward pass through chained `and` propagates correct gradients."""
    n = 4
    x = wp.array(np.ones(n, dtype=np.float32), device=device, requires_grad=True)
    flag = wp.array([1, 1, 0, 0], dtype=int, device=device)
    out = wp.zeros(n, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(test_short_circuit_and_grad_kernel, dim=n, inputs=[x, flag, out], device=device)

    # flag=1 and tid<2 → *3; else → *1
    np.testing.assert_allclose(out.numpy(), [3.0, 3.0, 1.0, 1.0])

    out.grad = wp.array(np.ones(n, dtype=np.float32), device=device)
    tape.backward()

    np.testing.assert_allclose(tape.gradients[x].numpy(), [3.0, 3.0, 1.0, 1.0])


def test_short_circuit_or_grad(test: unittest.TestCase, device):
    """Backward pass through chained `or` propagates correct gradients."""
    n = 4
    x = wp.array(np.ones(n, dtype=np.float32), device=device, requires_grad=True)
    flag = wp.array([0, 1, 1, 0], dtype=int, device=device)
    out = wp.zeros(n, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        wp.launch(test_short_circuit_or_grad_kernel, dim=n, inputs=[x, flag, out], device=device)

    # flag==0 or tid>=2 → *1; else → *3
    # tid=0: flag=0→true (short-circuit) → *1
    # tid=1: flag=1→false, tid>=2→false → *3
    # tid=2: flag=1→false, tid>=2→true  → *1
    # tid=3: flag=0→true (short-circuit) → *1
    np.testing.assert_allclose(out.numpy(), [1.0, 3.0, 1.0, 1.0])

    out.grad = wp.array(np.ones(n, dtype=np.float32), device=device)
    tape.backward()

    np.testing.assert_allclose(tape.gradients[x].numpy(), [1.0, 3.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Phi-elision in unrolled loops with predicate-guarded blocks (GH-1497).
#
# Codegen used to emit a ``wp::where`` phi at the end of every if-block for
# any reassigned symbol, including dead temporaries. With fully-unrolled
# loops this stacked into O(iters * temps) merge calls per kernel. The AST
# liveness pass in ``emit_If`` now skips phis for symbols that no downstream
# statement reads.
#
# These tests pin both the optimization (the GH-1497 reproducer matches a
# ``continue`` reference) and the cases it must NOT regress: loop-back
# accumulators, read-after-if temporaries, and ``if-else`` both-branch
# assignments. The gradient test pins adjoint correctness end-to-end.


@wp.kernel
def test_phi_elision_predicate_kernel(
    c_arr: wp.array2d(dtype=wp.float32),
    x_arr: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    # GH-1497 reproducer: predicate-guarded block whose temps a, b, d, v are
    # dead after the if. The inner loop unrolls fully.
    tid = wp.tid()
    max_val = wp.float(-1e20)
    for p in range(5):
        for q in range(5):
            c = c_arr[p, q]
            if c != 0.0:
                a = wp.float(p + 1)
                b = wp.float(q + 1)
                d = c * (a + b)
                v = x_arr[tid] * d
                if v > max_val:
                    max_val = v
    out[tid] = max_val


@wp.kernel
def test_phi_elision_continue_kernel(
    c_arr: wp.array2d(dtype=wp.float32),
    x_arr: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    # Equivalent control flow via ``continue``; the inner loop is dynamic
    # (Warp refuses to unroll loops containing ``continue``). Serves as the
    # bit-identical reference for the predicate form.
    tid = wp.tid()
    max_val = wp.float(-1e20)
    for p in range(5):
        for q in range(5):
            c = c_arr[p, q]
            if c == 0.0:
                continue
            a = wp.float(p + 1)
            b = wp.float(q + 1)
            d = c * (a + b)
            v = x_arr[tid] * d
            if v > max_val:
                max_val = v
    out[tid] = max_val


def test_phi_elision_predicate_matches_continue(test: unittest.TestCase, device):
    """Predicate form produces bit-identical output to the ``continue`` form."""
    rng = np.random.default_rng(0)
    n = 64
    for c_np in (
        np.zeros((5, 5), dtype=np.float32),
        rng.standard_normal((5, 5)).astype(np.float32),
        (rng.random((5, 5)) < 0.3).astype(np.float32) * rng.standard_normal((5, 5)).astype(np.float32),
        np.ones((5, 5), dtype=np.float32),
    ):
        x_np = rng.standard_normal(n).astype(np.float32)
        c_arr = wp.array(c_np, dtype=wp.float32, device=device)
        x_arr = wp.array(x_np, dtype=wp.float32, device=device)
        out_p = wp.zeros(n, dtype=wp.float32, device=device)
        out_c = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(test_phi_elision_predicate_kernel, n, inputs=[c_arr, x_arr, out_p], device=device)
        wp.launch(test_phi_elision_continue_kernel, n, inputs=[c_arr, x_arr, out_c], device=device)
        np.testing.assert_array_equal(out_p.numpy(), out_c.numpy())


@wp.kernel
def test_phi_elision_read_after_if_kernel(
    cond_arr: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # ``x`` is read after the if-block, so the phi must survive even though
    # ``x`` is introduced via reassignment inside the branch.
    x = wp.float(-1.0)
    if cond_arr[tid] != 0:
        x = wp.float(7.0)
    out[tid] = x


def test_phi_elision_read_after_if(test: unittest.TestCase, device):
    """Symbol read after the if must keep its phi merge."""
    cond = np.array([0, 1, 0, 1, 1, 0], dtype=np.int32)
    cw = wp.array(cond, device=device)
    ow = wp.zeros(cond.size, dtype=wp.float32, device=device)
    wp.launch(test_phi_elision_read_after_if_kernel, cond.size, inputs=[cw, ow], device=device)
    expected = np.where(cond != 0, 7.0, -1.0).astype(np.float32)
    np.testing.assert_array_equal(ow.numpy(), expected)


@wp.kernel
def test_phi_elision_accumulator_kernel(
    arr: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # Loop-back live: ``total`` is read-and-written inside the if, so the next
    # unrolled iteration depends on the previous iteration's phi.
    total = wp.float(0.0)
    for i in range(6):
        if arr[tid, i] > 0.0:
            total = total + arr[tid, i] * arr[tid, i]
    out[tid] = total


def test_phi_elision_loop_back_accumulator(test: unittest.TestCase, device):
    """Accumulator updated inside an unrolled-loop if must keep its phi."""
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((16, 6)).astype(np.float32)
    aw = wp.array(arr, device=device)
    ow = wp.zeros(arr.shape[0], dtype=wp.float32, device=device)
    wp.launch(test_phi_elision_accumulator_kernel, arr.shape[0], inputs=[aw, ow], device=device)
    expected = np.where(arr > 0.0, arr * arr, 0.0).sum(axis=1).astype(np.float32)
    np.testing.assert_allclose(ow.numpy(), expected, atol=1e-5)


@wp.kernel
def test_phi_elision_nested_if_kernel(
    arr: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # ``tmp`` is read only by the inner if, dead after the outer if — its phi
    # is eligible for elision. ``s`` is loop-back live and must survive.
    s = wp.float(0.0)
    for i in range(4):
        if arr[tid, i] > 0.0:
            tmp = arr[tid, i] * arr[tid, i]
            if tmp > 1.0:
                s = s + tmp
    out[tid] = s


def test_phi_elision_nested_if_dead_inner(test: unittest.TestCase, device):
    """Nested if where the outer-introduced temp is dead after its if-block."""
    rng = np.random.default_rng(2)
    arr = (rng.standard_normal((16, 4)) * 1.5).astype(np.float32)
    aw = wp.array(arr, device=device)
    ow = wp.zeros(arr.shape[0], dtype=wp.float32, device=device)
    wp.launch(test_phi_elision_nested_if_kernel, arr.shape[0], inputs=[aw, ow], device=device)
    sq = arr * arr
    expected = np.where((arr > 0.0) & (sq > 1.0), sq, 0.0).sum(axis=1).astype(np.float32)
    np.testing.assert_allclose(ow.numpy(), expected, atol=1e-5)


@wp.kernel
def test_phi_elision_if_else_kernel(
    c: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    # Same symbol assigned in both branches: the body-phi loop registers
    # ``v``, then the else-phi loop emits the merge because ``v`` is live-out.
    if c[tid] > 0:
        v = wp.float(1.0)
    else:
        v = wp.float(-2.0)
    out[tid] = v


def test_phi_elision_if_else_both_branches(test: unittest.TestCase, device):
    """``if-else`` where the same symbol is assigned in both branches."""
    c = np.array([-2, 0, 1, 5, -3, 2], dtype=np.int32)
    cw = wp.array(c, device=device)
    ow = wp.zeros(c.size, dtype=wp.float32, device=device)
    wp.launch(test_phi_elision_if_else_kernel, c.size, inputs=[cw, ow], device=device)
    expected = np.where(c > 0, 1.0, -2.0).astype(np.float32)
    np.testing.assert_array_equal(ow.numpy(), expected)


def test_phi_elision_predicate_grad(test: unittest.TestCase, device):
    """Adjoint of the unrolled predicate form matches an analytical reference.

    The gradient of ``max_val`` w.r.t. ``x_arr[tid]`` equals ``d`` at the
    argmax (since ``v = x_arr[tid] * d``). If phi-elision incorrectly dropped
    a live merge, the argmax tracking would break and gradients would differ
    from the reference computed by direct iteration in NumPy.
    """
    rng = np.random.default_rng(3)
    n = 16
    c_np = (rng.random((5, 5)) < 0.5).astype(np.float32) * rng.standard_normal((5, 5)).astype(np.float32)
    x_np = rng.standard_normal(n).astype(np.float32) * 1.7

    # Hand-compute the gradient: at the argmax (p*, q*), d/dx[tid] of max_val
    # is just d* = c[p*, q*] * ((p*+1) + (q*+1)).
    ref_grad = np.zeros(n, dtype=np.float32)
    for tid in range(n):
        best_v = -1e20
        best_d = np.float32(0.0)
        for p in range(5):
            for q in range(5):
                c = c_np[p, q]
                if c != 0.0:
                    d = c * np.float32((p + 1) + (q + 1))
                    v = x_np[tid] * d
                    if v > best_v:
                        best_v = v
                        best_d = d
        ref_grad[tid] = best_d

    x = wp.array(x_np, device=device, requires_grad=True)
    c = wp.array(c_np, device=device, requires_grad=False)
    out = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(test_phi_elision_predicate_kernel, n, inputs=[c, x, out], device=device)
    out.grad = wp.array(np.ones(n, dtype=np.float32), device=device)
    tape.backward()
    np.testing.assert_array_equal(tape.gradients[x].numpy(), ref_grad)


devices = get_test_devices()


class TestConditional(unittest.TestCase):
    pass


add_kernel_test(TestConditional, kernel=test_conditional_if_else, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_if_else_nested, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_ifexp, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_ifexp_nested, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_ifexp_constant, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_ifexp_constant_nested, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_and, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_or, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_compound, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_literal, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_int_logical_not, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_int_conditional_assign_overload, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_bool_param_conditional, dim=1, inputs=[True], devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_basic, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_empty_range, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_faker, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_and, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_eqs, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_mixed, dim=1, devices=devices)
add_function_test(TestConditional, "test_conditional_unequal_types", test_conditional_unequal_types, devices=devices)
add_function_test(TestConditional, "test_ifexp_with_array_access", test_ifexp_with_array_access, devices=devices)
add_function_test(TestConditional, "test_short_circuit_and", test_short_circuit_and, devices=devices)
add_function_test(TestConditional, "test_short_circuit_or", test_short_circuit_or, devices=devices)
add_function_test(TestConditional, "test_short_circuit_and_grad", test_short_circuit_and_grad, devices=devices)
add_function_test(TestConditional, "test_short_circuit_or_grad", test_short_circuit_or_grad, devices=devices)
add_function_test(
    TestConditional,
    "test_phi_elision_predicate_matches_continue",
    test_phi_elision_predicate_matches_continue,
    devices=devices,
)
add_function_test(
    TestConditional,
    "test_phi_elision_read_after_if",
    test_phi_elision_read_after_if,
    devices=devices,
)
add_function_test(
    TestConditional,
    "test_phi_elision_loop_back_accumulator",
    test_phi_elision_loop_back_accumulator,
    devices=devices,
)
add_function_test(
    TestConditional,
    "test_phi_elision_nested_if_dead_inner",
    test_phi_elision_nested_if_dead_inner,
    devices=devices,
)
add_function_test(
    TestConditional,
    "test_phi_elision_if_else_both_branches",
    test_phi_elision_if_else_both_branches,
    devices=devices,
)
add_function_test(
    TestConditional,
    "test_phi_elision_predicate_grad",
    test_phi_elision_predicate_grad,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
