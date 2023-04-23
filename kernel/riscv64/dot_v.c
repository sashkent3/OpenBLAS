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

#if !defined(DOUBLE)
#define FLOAT_V_T_M1 vfloat32m1_t
#define FLOAT_V_T vfloat32m8_t
#define VL_M1(ptr, vl) __riscv_vle32_v_f32m1(ptr, vl)
#define VLS(ptr, stride, vl) __riscv_vlse32_v_f32m8(ptr, stride, vl)
#define VSETVL(n) __riscv_vsetvl_e32m8(n)
#define VFMUL(x, y, vl) __riscv_vfmul_vv_f32m8(x, y, vl)
#define VFREDSUM(x, s, vl) __riscv_vfredusum_vs_f32m8_f32m1(x, s, vl)
#define VSE_M1(ptr, x, vl) __riscv_vse32_v_f32m1(ptr, x, vl)
#else
#define FLOAT_V_T_M1 vfloat64m1_t
#define FLOAT_V_T vfloat64m8_t
#define VL_M1(ptr, vl) __riscv_vle64_v_f64m1(ptr, vl)
#define VLS(ptr, stride, vl) __riscv_vlse64_v_f64m8(ptr, stride, vl)
#define VSETVL(n) __riscv_vsetvl_e64m8(n)
#define VFMUL(x, y, vl) __riscv_vfmul_vv_f64m8(x, y, vl)
#define VFREDSUM(x, s, vl) __riscv_vfredusum_vs_f64m8_f64m1(x, s, vl)
#define VSE_M1(ptr, x, vl) __riscv_vse64_v_f64m1(ptr, x, vl)
#endif

#if defined(DSDOT)
double CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y)
#else
FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y)
#endif
{
	const FLOAT zero = 0;
    FLOAT_V_T_M1 zero_v = VL_M1(&zero, 1);
    size_t vl;
    FLOAT dot = 0, chunk;
    for (BLASLONG offset = 0; offset < n; offset += vl)
    {
        vl = VSETVL(n - offset);
        FLOAT_V_T x_v = VLS(x + offset * inc_x, inc_x * sizeof(FLOAT), vl);
        FLOAT_V_T y_v = VLS(y + offset * inc_y, inc_y * sizeof(FLOAT), vl);
        FLOAT_V_T prod_v = VFMUL(x_v, y_v, vl);
        FLOAT_V_T_M1 dot_v = VFREDSUM(prod_v, zero_v, vl);
        VSE_M1(&chunk, dot_v, 1);
        dot += chunk;
    }
    return dot;
}


