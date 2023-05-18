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

#ifdef DOUBLE
#define VFLOAT_T vfloat64m8_t
#define VFLOAT_T_M1 vfloat64m1_t
#define VL __riscv_vle64_v_f64m8
#define VLS __riscv_vlse64_v_f64m8
#define VSETVL __riscv_vsetvl_e64m8
#define VFMAX __riscv_vfmax_vv_f64m8
#define VFMV_S_F __riscv_vfmv_s_f_f64m1
#define VFREDMAX __riscv_vfredmax_vs_f64m8_f64m1
#define VFMV_F_S __riscv_vfmv_f_s_f64m1_f64
#else
#define VFLOAT_T vfloat32m8_t
#define VFLOAT_T_M1 vfloat32m1_t
#define VL __riscv_vle32_v_f32m8
#define VLS __riscv_vlse32_v_f32m8
#define VSETVL __riscv_vsetvl_e32m8
#define VFMAX __riscv_vfmax_vv_f32m8
#define VFMV_S_F __riscv_vfmv_s_f_f32m1
#define VFREDMAX __riscv_vfredmax_vs_f32m8_f32m1
#define VFMV_F_S __riscv_vfmv_f_s_f32m1_f32
#endif

FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x) {
    if (n <= 0 || inc_x <= 0) return 0.0;

    VFLOAT_T_M1 res_v = VFMV_S_F(*x, 1);
    x += inc_x; --n;
    size_t vl_start = VSETVL(n), vl;
    VFLOAT_T chunk_v, x_v;
    if (inc_x == 1) {
        chunk_v = VL(x, vl_start);
        for (; vl_start < n; n -= vl, x += vl) {
            vl = VSETVL(n);
            x_v = VL(x, vl);
            chunk_v = VFMAX(chunk_v, x_v, vl);
        }
    } else {
        ptrdiff_t stride_x = inc_x * sizeof(FLOAT);
        chunk_v = VLS(x, stride_x, vl_start);
        for (; vl_start < n; n -= vl, x += vl * inc_x) {
            vl = VSETVL(n);
            x_v = VLS(x, stride_x, vl);
            chunk_v = VFMAX(chunk_v, x_v, vl);
        }
    } 
    res_v = VFREDMAX(chunk_v, res_v, vl_start);
    return VFMV_F_S(res_v);
}
