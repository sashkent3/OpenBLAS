/***************************************************************************
Copyright (c) 2023, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

#include "common.h"
#include <math.h>

#ifdef DOUBLE
#define VFLOAT_T vfloat64m8_t
#define VFLOAT_T_M1 vfloat64m1_t
#define VL __riscv_vle64_v_f64m8
#define VLS __riscv_vlse64_v_f64m8
#define VSETVL __riscv_vsetvl_e64m8
#define VFABS __riscv_vfabs_v_f64m8
#define VFDIV_F __riscv_vfdiv_vf_f64m8
#define VFMUL_F __riscv_vfmul_vf_f64m8
#define VFADD_V __riscv_vfadd_vv_f64m8
#define VFMV_S_F __riscv_vfmv_s_f_f64m1
#define VFMV_V_F __riscv_vfmv_v_f_f64m8
#define VFREDMAX __riscv_vfredmax_vs_f64m8_f64m1
#define VFREDSUM __riscv_vfredusum_vs_f64m8_f64m1
#define VFMV_F_S __riscv_vfmv_f_s_f64m1_f64
#define VFMACC __riscv_vfmacc_vv_f64m8
#define FABS fabs
#define SQRT sqrt
#else
#define VFLOAT_T vfloat32m8_t
#define VFLOAT_T_M1 vfloat32m1_t
#define VL __riscv_vle32_v_f32m8
#define VLS __riscv_vlse32_v_f32m8
#define VSETVL __riscv_vsetvl_e32m8
#define VFABS __riscv_vfabs_v_f32m8
#define VFDIV_F __riscv_vfdiv_vf_f32m8
#define VFMUL_F __riscv_vfmul_vf_f32m8
#define VFADD_V __riscv_vfadd_vv_f32m8
#define VFMV_S_F __riscv_vfmv_s_f_f32m1
#define VFMV_V_F __riscv_vfmv_v_f_f32m8
#define VFREDMAX __riscv_vfredmax_vs_f32m8_f32m1
#define VFREDSUM __riscv_vfredusum_vs_f32m8_f32m1
#define VFMV_F_S __riscv_vfmv_f_s_f32m1_f32
#define VFMACC __riscv_vfmacc_vv_f32m8
#define FABS fabsf
#define SQRT sqrtf
#endif

FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x) {
    if (n <= 0) return 0.0;
    FLOAT first = FABS(*x);
    if (n == 1) return first;
    if (inc_x == 0) return SQRT((FLOAT) n) * first;
    FLOAT scale = first, scale_new, rescale_coef;
    VFLOAT_T_M1 scale_v = VFMV_S_F(scale, 1);
    x += inc_x; --n;
    size_t vl_start = VSETVL(n);
    VFLOAT_T ssq_v = VFMV_V_F(0.0, vl_start), x_v;
    size_t vl;
    if (inc_x == 1) {
        for (; 0 < n; n -= vl, x += vl) {
            vl = VSETVL(n);
            x_v = VL(x, vl);
            x_v = VFABS(x_v, vl);
            scale_v = VFREDMAX(x_v, scale_v, vl);
            scale_new = VFMV_F_S(scale_v);
            switch (fpclassify(scale_new)) {
                case FP_ZERO: continue;
                case FP_NAN:
                case FP_INFINITE: return scale_new;
            }
            if (scale_new > scale) {
                rescale_coef = scale / scale_new;
                ssq_v = VFMUL_F(ssq_v, rescale_coef * rescale_coef, vl_start);
            }
            scale = scale_new;
            x_v = VFDIV_F(x_v, scale, vl);
            ssq_v = VFMACC(ssq_v, x_v, x_v, vl);
        }
    } else {
        ptrdiff_t stride_x = inc_x * sizeof(FLOAT);
        for (; 0 < n; n -= vl, x += vl) {
            vl = VSETVL(n);
            x_v = VLS(x, stride_x, vl);
            x_v = VFABS(x_v, vl);
            scale_v = VFREDMAX(x_v, scale_v, vl);
            scale_new = VFMV_F_S(scale_v);
            switch (fpclassify(scale_new)) {
                case FP_ZERO: continue;
                case FP_NAN:
                case FP_INFINITE: return scale_new;
            }
            if (scale_new > scale) {
                rescale_coef = scale / scale_new;
                ssq_v = VFMUL_F(ssq_v, rescale_coef * rescale_coef, vl_start);
            }
            scale = scale_new;
            x_v = VFDIV_F(x_v, scale, vl);
            ssq_v = VFMACC(ssq_v, x_v, x_v, vl);
        }
    }
    if (scale == 0) return 0.0;
    first /= scale;
    FLOAT ssq = VFMV_F_S(VFREDSUM(ssq_v, VFMV_S_F(first * first, 1), vl_start));
    return scale * SQRT(ssq);
}
