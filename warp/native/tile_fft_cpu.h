// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#include "vec.h"

// CPU implementation of tile_fft / tile_ifft.
//
// The tile API dispatches FFTs to cuFFTDx on device; on the host there is no
// equivalent library so we provide a self-contained implementation here.
// Input is a register tile of vec2<T> (complex) entries laid out row-major in
// the flat `data` array, with the FFT running over the last (innermost) axis
// and all leading axes treated as independent batches.
//
// Algorithms:
//   * radix-2 iterative Cooley-Tukey for power-of-two sizes (O(N log N))
//   * direct DFT fallback for other sizes (O(N^2))
//
// Both paths operate in place and match cuFFTDx's unnormalized convention:
//   forward: X_k = sum_n x_n * exp(-2*pi*i*k*n/N)
//   inverse: X_k = sum_n x_n * exp(+2*pi*i*k*n/N)  (no 1/N scaling)

namespace wp {

namespace fft_cpu_detail {

template <typename T> struct fft_consts {
    static constexpr T two_pi() { return T(6.28318530717958647692528676655900576839433879875021); }
};

// returns log2(n) if n > 0 is a power of two, otherwise -1
inline CUDA_CALLABLE int fft_log2_pow2(int n)
{
    if (n <= 0)
        return -1;
    int log_n = 0;
    int x = n;
    while ((x & 1) == 0) {
        x >>= 1;
        ++log_n;
    }
    return (x == 1) ? log_n : -1;
}

// in-place bit-reverse permutation for size n (n must be power of 2)
template <typename T> inline CUDA_CALLABLE void fft_bit_reverse_permute(vec_t<2, T>* x, int n)
{
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            vec_t<2, T> tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
}

// In-place radix-2 iterative Cooley-Tukey FFT of size n (n power of 2).
// direction_sign selects forward (-1) or inverse (+1) — i.e. the sign of the
// exponent in the DFT kernel. Unnormalized in both directions.
template <typename T> inline CUDA_CALLABLE void fft_radix2_inplace(vec_t<2, T>* x, int n, int direction_sign)
{
    fft_bit_reverse_permute<T>(x, n);

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const T theta = T(direction_sign) * fft_consts<T>::two_pi() / T(len);
        const T wm_re = wp::cos(theta);
        const T wm_im = wp::sin(theta);

        for (int k = 0; k < n; k += len) {
            T w_re = T(1);
            T w_im = T(0);
            for (int j = 0; j < half; ++j) {
                const T x_re = x[k + j + half].c[0];
                const T x_im = x[k + j + half].c[1];
                const T t_re = w_re * x_re - w_im * x_im;
                const T t_im = w_re * x_im + w_im * x_re;

                const T u_re = x[k + j].c[0];
                const T u_im = x[k + j].c[1];

                x[k + j].c[0] = u_re + t_re;
                x[k + j].c[1] = u_im + t_im;
                x[k + j + half].c[0] = u_re - t_re;
                x[k + j + half].c[1] = u_im - t_im;

                const T new_w_re = w_re * wm_re - w_im * wm_im;
                const T new_w_im = w_re * wm_im + w_im * wm_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
        }
    }
}

// Maximum supported FFT size for the non-power-of-2 fallback.
// The fallback uses stack scratch proportional to this, keep it modest.
#define WP_FFT_CPU_MAX_DFT_SIZE 4096

// Naive O(N^2) DFT fallback for non-power-of-two sizes.
// Uses a small local scratch buffer on the stack; callers bound size.
template <typename T> inline CUDA_CALLABLE void fft_dft_inplace(vec_t<2, T>* x, int n, int direction_sign)
{
    assert(n <= WP_FFT_CPU_MAX_DFT_SIZE);

    T out_re[WP_FFT_CPU_MAX_DFT_SIZE];
    T out_im[WP_FFT_CPU_MAX_DFT_SIZE];

    const T base = T(direction_sign) * fft_consts<T>::two_pi() / T(n);

    for (int k = 0; k < n; ++k) {
        T acc_re = T(0);
        T acc_im = T(0);
        for (int j = 0; j < n; ++j) {
            const T theta = base * T(k) * T(j);
            const T c = wp::cos(theta);
            const T s = wp::sin(theta);
            const T xr = x[j].c[0];
            const T xi = x[j].c[1];
            acc_re += c * xr - s * xi;
            acc_im += c * xi + s * xr;
        }
        out_re[k] = acc_re;
        out_im[k] = acc_im;
    }

    for (int k = 0; k < n; ++k) {
        x[k].c[0] = out_re[k];
        x[k].c[1] = out_im[k];
    }
}

// Dispatches a single-batch in-place FFT to the fastest applicable kernel.
template <typename T> inline CUDA_CALLABLE void fft_inplace(vec_t<2, T>* x, int n, int direction_sign)
{
    if (fft_log2_pow2(n) >= 0) {
        fft_radix2_inplace<T>(x, n, direction_sign);
    } else {
        fft_dft_inplace<T>(x, n, direction_sign);
    }
}

}  // namespace fft_cpu_detail

namespace fft_cpu_detail {
// Extract the component type from a wp::vec_t<2, T>. Only the vec2 case is
// used by tile_fft, but a generic trait keeps the macro signature clean.
template <typename V> struct vec2_component;
template <typename T> struct vec2_component<vec_t<2, T>> {
    using type = T;
};
}  // namespace fft_cpu_detail

// direction_sign: -1 = forward, +1 = inverse (both unnormalized)
template <int DirectionSign, typename Complex>
inline CUDA_CALLABLE void tile_fft_cpu_impl(int batch, int fft_size, Complex* data)
{
    using T = typename fft_cpu_detail::vec2_component<Complex>::type;
    for (int b = 0; b < batch; ++b) {
        fft_cpu_detail::fft_inplace<T>(data + b * fft_size, fft_size, DirectionSign);
    }
}

}  // namespace wp
