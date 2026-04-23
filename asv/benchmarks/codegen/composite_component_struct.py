# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ASV benchmarks for composite-component writes on ``wp.array(dtype=StateStruct)``.

``StateStruct`` is a representative physics-simulation state: two ``vec3``
fields plus a ``quat`` field plus a scalar. 12 + 12 + 16 + 4 = 44 bytes.
The "multi-slot write" benchmark mirrors a typical per-particle update step.

These benchmarks exercise the pattern that motivated the original
``enable_vector_component_overwrites`` compile-time concern: kernels that
write many composite slots per element. Any lowering that over-produces IR
per write shows up here.
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


@wp.kernel
def state_multi_field(
    dst: wp.array(dtype=StateStruct),
    p: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quatf),
    m: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dst[i].position = p[i]
    dst[i].velocity = v[i]
    dst[i].rotation = q[i]
    dst[i].mass = m[i]


@wp.kernel
def state_augassign_field(dst: wp.array(dtype=StateStruct), p: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    dst[i].position += p[i]


@wp.kernel
def state_full_element(dst: wp.array(dtype=StateStruct), src: wp.array(dtype=StateStruct)):
    i = wp.tid()
    dst[i] = src[i]


FWD_LAUNCHES = 1000
BWD_LAUNCHES = 1000


class CompileSingleField:
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


class CompileMultiField:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        state_multi_field.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class CompileAugassignField:
    repeat = 10
    number = 1

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        state_augassign_field.module.unload()
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
        state_full_element.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class RunForwardSingleField:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0")
        self.p = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0")
        self.cmd = wp.launch(state_single_field, self.n, inputs=[self.dst, self.p], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunForwardMultiField:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0")
        self.p = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0")
        self.v = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0")
        self.q = wp.zeros(self.n, dtype=wp.quatf, device="cuda:0")
        self.m = wp.ones(self.n, dtype=wp.float32, device="cuda:0")
        self.cmd = wp.launch(
            state_multi_field,
            self.n,
            inputs=[self.dst, self.p, self.v, self.q, self.m],
            device="cuda:0",
            record_cmd=True,
        )
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
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0")
        self.src = wp.zeros(self.n, dtype=StateStruct, device="cuda:0")
        self.cmd = wp.launch(state_full_element, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(FWD_LAUNCHES):
                self.cmd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardSingleField:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0", requires_grad=True)
        self.p = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
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


class RunBackwardMultiField:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 18
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0", requires_grad=True)
        self.p = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        self.v = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        self.q = wp.zeros(self.n, dtype=wp.quatf, device="cuda:0", requires_grad=True)
        self.m = wp.ones(self.n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    state_multi_field,
                    self.n,
                    inputs=[self.dst, self.p, self.v, self.q, self.m],
                    adj_inputs=[self.dst.grad, self.p.grad, self.v.grad, self.q.grad, self.m.grad],
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
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0", requires_grad=True)
        self.src = wp.zeros(self.n, dtype=StateStruct, device="cuda:0", requires_grad=True)
        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(BWD_LAUNCHES):
                wp.launch(
                    state_full_element,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"
