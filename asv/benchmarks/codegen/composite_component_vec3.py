# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASV benchmarks for composite-component writes on ``wp.array(dtype=wp.vec3)``.

Covers single-slot write, multi-slot-per-element writes, augassign, and the
whole-element-write baseline. Tracks compile time (CUDA codegen + NVRTC),
forward-only kernel runtime, and forward+backward kernel runtime.

Used to detect regressions from changes to the composite-component write
lowering (the silent-zero-gradient adjoint fix and its successors). The
whole-element-write baseline gives a reference for "cost of one
``array_store`` of a vec3 element" — any slot-write lowering that
approaches this is not regressing the element-write pattern the hardware
already does efficiently.
"""

from statistics import median

import warp as wp

from ..benchmarks_utils import clear_kernel_cache

# ---- kernels under test ----


@wp.kernel
def vec3_single_slot(dst: wp.array(dtype=wp.vec3), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i].y = src[i]


@wp.kernel
def vec3_triple_slot(dst: wp.array(dtype=wp.vec3), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i].x = src[i] * 2.0
    dst[i].y = src[i] * 3.0
    dst[i].z = src[i] * 4.0


@wp.kernel
def vec3_augassign_slot(dst: wp.array(dtype=wp.vec3), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i].y += src[i]


@wp.kernel
def vec3_full_element(dst: wp.array(dtype=wp.vec3), src: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i] = src[i]


# Helper: how many launches to time. More launches = tighter variance; tune
# to keep individual benchmark run under a few seconds.
FWD_LAUNCHES = 1000
BWD_LAUNCHES = 1000


def _make_grad_arrays(n):
    dst = wp.zeros(n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    src = wp.ones(n, dtype=wp.float32, device="cuda:0", requires_grad=True)
    return dst, src


def _make_grad_arrays_vec(n):
    dst = wp.zeros(n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    src = wp.ones(n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
    return dst, src


# ---- compile-time benchmarks ----


class CompileSingleSlot:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        vec3_single_slot.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class CompileTripleSlot:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        vec3_triple_slot.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class CompileAugassign:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        vec3_augassign_slot.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class CompileFullElement:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        vec3_full_element.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


# ---- forward-only runtime benchmarks ----


class RunForwardSingleSlot:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0")
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0")
        self.cmd = wp.launch(vec3_single_slot, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunForwardTripleSlot:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0")
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0")
        self.cmd = wp.launch(vec3_triple_slot, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunForwardAugassign:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0")
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0")
        self.cmd = wp.launch(vec3_augassign_slot, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunForwardFullElement:
    """Baseline: whole-element overwrite — the cheapest array_store pattern."""

    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0")
        self.src = wp.ones(self.n, dtype=wp.vec3, device="cuda:0")
        self.cmd = wp.launch(vec3_full_element, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


# ---- forward+backward runtime benchmarks ----


class RunBackwardSingleSlot:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst, self.src = _make_grad_arrays(self.n)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    vec3_single_slot,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardTripleSlot:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst, self.src = _make_grad_arrays(self.n)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    vec3_triple_slot,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardAugassign:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst, self.src = _make_grad_arrays(self.n)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    vec3_augassign_slot,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardFullElement:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst, self.src = _make_grad_arrays_vec(self.n)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    vec3_full_element,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"
