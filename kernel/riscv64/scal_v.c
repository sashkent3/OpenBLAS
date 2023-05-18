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
#define VL __riscv_vle64_v_f64m8
#define VLS __riscv_vlse64_v_f64m8
#define VSETVL __riscv_vsetvl_e64m8
#define VFMUL __riscv_vfmul_vf_f64m8
#define VS __riscv_vse64_v_f64m8
#define VSS __riscv_vsse64_v_f64m8
#else
#define VFLOAT_T vfloat32m8_t
#define VL __riscv_vle32_v_f32m8
#define VLS __riscv_vlse32_v_f32m8
#define VSETVL __riscv_vsetvl_e32m8
#define VFMUL __riscv_vfmul_vf_f32m8
#define VS __riscv_vse32_v_f32m8
#define VSS __riscv_vsse32_v_f32m8
#endif

int CNAME(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, FLOAT da, FLOAT *x, BLASLONG inc_x, FLOAT *y, BLASLONG inc_y, FLOAT *dummy, BLASLONG dummy2) {
    size_t vl;
    VFLOAT_T x_v;
    if (inc_x == 1) {
        for (; 0 < n; n -= vl, x += vl) {
            vl = VSETVL(n);
            x_v = VL(x, vl);
            x_v = VFMUL(x_v, da, vl);
            VS(x, x_v, vl);
        }
    } else {
        ptrdiff_t stride_x = inc_x * sizeof(FLOAT);
        for (; 0 < n; n -= vl, x += vl * inc_x) {
            vl = VSETVL(n);
            x_v = VLS(x, stride_x, vl);
            x_v = VFMUL(x_v, da, vl);
            VSS(x, stride_x, x_v, vl);
        }
    }
}
