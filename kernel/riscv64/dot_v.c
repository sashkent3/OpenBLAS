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

#if defined(DSDOT)
#define VFMACC __riscv_vfwmacc_vv_f64m8
#elif defined(DOUBLE)
#define VFMACC __riscv_vfmacc_vv_f64m8
#else
#define VFMACC __riscv_vfmacc_vv_f32m4
#endif

#if defined(DOUBLE)
#define VFLOAT_T vfloat64m8_t
#define VL __riscv_vle64_v_f64m8
#define VLS __riscv_vlse64_v_f64m8
#define VSETVL __riscv_vsetvl_e64m8
#else
#define VFLOAT_T vfloat32m4_t
#define VL __riscv_vle32_v_f32m4
#define VLS __riscv_vlse32_v_f32m4
#define VSETVL __riscv_vsetvl_e32m4
#endif

#if defined(DSDOT) || defined(DOUBLE)
#define VFLOAT_RES_T_M1 vfloat64m1_t
#define VFLOAT_RES_T vfloat64m8_t
#define VFREDSUM __riscv_vfredusum_vs_f64m8_f64m1
#define VFMV_V_F __riscv_vfmv_v_f_f64m8
#define VFMV_S_F __riscv_vfmv_s_f_f64m1
#define VFMV_F_S __riscv_vfmv_f_s_f64m1_f64
#else
#define VFLOAT_RES_T_M1 vfloat32m1_t
#define VFLOAT_RES_T vfloat32m4_t
#define VFREDSUM __riscv_vfredusum_vs_f32m4_f32m1
#define VFMV_V_F __riscv_vfmv_v_f_f32m4
#define VFMACC __riscv_vfmacc_vv_f32m4
#define VFMV_S_F __riscv_vfmv_s_f_f32m1
#define VFMV_F_S __riscv_vfmv_f_s_f32m1_f32
#endif

#if defined(DSDOT)
double CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y)
#else
FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y)
#endif
{
    if (n <= 0) return 0;
    VFLOAT_RES_T_M1 res_v = VFMV_S_F(*x * *y, 1);
    --n;
    size_t vl_start = VSETVL(n), vl;
    x += inc_x;
    y += inc_y;
    VFLOAT_RES_T res_chunks_v = VFMV_V_F(0, vl_start);
    VFLOAT_T x_v, y_v;
    if (inc_x == 1) {
        if (inc_y == 1) {
            for (BLASLONG offset = 0; offset < n; offset += vl) {
                vl = VSETVL(n - offset);
                x_v = VL(x + offset, vl);
                y_v = VL(y + offset, vl);
                res_chunks_v = VFMACC(res_chunks_v, x_v, y_v, vl);
            }
        } else {
            ptrdiff_t stride_y = inc_y * sizeof(FLOAT);
            for (BLASLONG offset = 0; offset < n; offset += vl) {
                vl = VSETVL(n - offset);
                x_v = VL(x + offset, vl);
                y_v = VLS(y + offset * inc_y, stride_y, vl);
                res_chunks_v = VFMACC(res_chunks_v, x_v, y_v, vl);
            }
        }
    } else {
        ptrdiff_t stride_x = inc_x * sizeof(FLOAT);
        if (inc_y == 1) {
            for (BLASLONG offset = 0; offset < n; offset += vl) {
                vl = VSETVL(n - offset);
                x_v = VLS(x + offset * inc_x, stride_x, vl);
                y_v = VL(y + offset, vl);
                res_chunks_v = VFMACC(res_chunks_v, x_v, y_v, vl);
            }
        } else {
            ptrdiff_t stride_y = inc_y * sizeof(FLOAT);
            for (BLASLONG offset = 0; offset < n; offset += vl) {
                vl = VSETVL(n - offset);
                x_v = VLS(x + offset * inc_x, stride_x, vl);
                y_v = VLS(y + offset * inc_y, stride_y, vl);
                res_chunks_v = VFMACC(res_chunks_v, x_v, y_v, vl);
            }
        }
    }
    res_v = VFREDSUM(res_chunks_v, res_v, vl_start);
    return VFMV_F_S(res_v);
}
