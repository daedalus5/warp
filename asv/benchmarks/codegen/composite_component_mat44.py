# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASV benchmarks for composite-component writes on ``wp.array(dtype=wp.mat44)``.

``mat44`` is a 64-byte composite — large enough that a whole-element
load/store per slot write has measurably different memory traffic than a
direct single-slot store. These benchmarks put a floor under how costly
any lowering can be for matrix element writes before it becomes user-visible.
"""

from statistics import median

import warp as wp

from ..benchmarks_utils import clear_kernel_cache


@wp.kernel
def mat44_single_elem(dst: wp.array(dtype=wp.mat44), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i][1, 2] = src[i]


@wp.kernel
def mat44_multi_elem(dst: wp.array(dtype=wp.mat44), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    # Simulate constraint-jacobian row population: write several entries.
    dst[i][0, 0] = src[i] * 1.0
    dst[i][0, 3] = src[i] * 2.0
    dst[i][1, 1] = src[i] * 3.0
    dst[i][2, 2] = src[i] * 4.0
    dst[i][3, 3] = src[i] * 5.0


@wp.kernel
def mat44_augassign_elem(dst: wp.array(dtype=wp.mat44), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i][1, 2] += src[i]


@wp.kernel
def mat44_full_element(dst: wp.array(dtype=wp.mat44), src: wp.array(dtype=wp.mat44)):
    i = wp.tid()
    dst[i] = src[i]


FWD_LAUNCHES = 1000
BWD_LAUNCHES = 1000


class CompileSingleElem:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        mat44_single_elem.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class CompileMultiElem:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        mat44_multi_elem.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class CompileAugassignElem:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        mat44_augassign_elem.module.unload()
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
        mat44_full_element.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class RunForwardSingleElem:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=wp.mat44, device="cuda:0")
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0")
        self.cmd = wp.launch(mat44_single_elem, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunForwardMultiElem:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=wp.mat44, device="cuda:0")
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0")
        self.cmd = wp.launch(mat44_multi_elem, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunForwardFullElement:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=wp.mat44, device="cuda:0")
        self.src = wp.ones(self.n, dtype=wp.mat44, device="cuda:0")
        self.cmd = wp.launch(mat44_full_element, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardSingleElem:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=wp.mat44, device="cuda:0", requires_grad=True)
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    mat44_single_elem,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardMultiElem:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=wp.mat44, device="cuda:0", requires_grad=True)
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    mat44_multi_elem,
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
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=wp.mat44, device="cuda:0", requires_grad=True)
        self.src = wp.ones(self.n, dtype=wp.mat44, device="cuda:0", requires_grad=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    mat44_full_element,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"
