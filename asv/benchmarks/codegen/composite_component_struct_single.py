# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single struct-field write on a user ``@wp.struct`` array: updating one
composite-valued field (``wp.vec3``) of a larger struct. Exercises the
struct-field branch of the fast path.
"""

from statistics import median

import warp as wp

from ..benchmarks_utils import clear_kernel_cache


@wp.struct
class StateStruct:
    position: wp.vec3
    velocity: wp.vec3
    rotation: wp.quatf
    mass: wp.float32


@wp.kernel
def state_single_field(dst: wp.array(dtype=StateStruct), p: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i].position = p[i]


class CompileModule:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        state_single_field.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class RunForwardKernel:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0")
        self.p = wp.ones(self.n, dtype=wp.vec3, device="cuda:0")
        self.cmd = wp.launch(state_single_field, self.n, inputs=[self.dst, self.p], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardKernel:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0", requires_grad=True)
        self.p = wp.ones(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                wp.launch(
                    state_single_field,
                    self.n,
                    inputs=[self.dst, self.p],
                    adj_inputs=[self.dst.grad, self.p.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"
