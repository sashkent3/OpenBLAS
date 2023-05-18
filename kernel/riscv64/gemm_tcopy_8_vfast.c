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
#define VFLOAT_T vfloat64m1_t
#define VSETVL __riscv_vsetvl_e64m1
#define VLSSEG8 __riscv_vlsseg8e64_v_f64m1
#define VLSSEG7 __riscv_vlsseg7e64_v_f64m1
#define VLSSEG6 __riscv_vlsseg6e64_v_f64m1
#define VLSSEG5 __riscv_vlsseg5e64_v_f64m1
#define VLSSEG4 __riscv_vlsseg4e64_v_f64m1
#define VLSSEG3 __riscv_vlsseg3e64_v_f64m1
#define VLSSEG2 __riscv_vlsseg2e64_v_f64m1
#define VLS __riscv_vlse64_v_f64m1
#define VSSEG8 __riscv_vsseg8e64_v_f64m1
#define VSSEG7 __riscv_vsseg7e64_v_f64m1
#define VSSEG6 __riscv_vsseg6e64_v_f64m1
#define VSSEG5 __riscv_vsseg5e64_v_f64m1
#define VSSEG4 __riscv_vsseg4e64_v_f64m1
#define VSSEG3 __riscv_vsseg3e64_v_f64m1
#define VSSEG2 __riscv_vsseg2e64_v_f64m1
#define VS __riscv_vse64_v_f64m1
#else
#define VFLOAT_T vfloat32m1_t
#define VSETVL __riscv_vsetvl_e32m1
#define VLSSEG8 __riscv_vlsseg8e32_v_f32m1
#define VLSSEG7 __riscv_vlsseg7e32_v_f32m1
#define VLSSEG6 __riscv_vlsseg6e32_v_f32m1
#define VLSSEG5 __riscv_vlsseg5e32_v_f32m1
#define VLSSEG4 __riscv_vlsseg4e32_v_f32m1
#define VLSSEG3 __riscv_vlsseg3e32_v_f32m1
#define VLSSEG2 __riscv_vlsseg2e32_v_f32m1
#define VLS __riscv_vlse32_v_f32m1
#define VSSEG8 __riscv_vsseg8e32_v_f32m1
#define VSSEG7 __riscv_vsseg7e32_v_f32m1
#define VSSEG6 __riscv_vsseg6e32_v_f32m1
#define VSSEG5 __riscv_vsseg5e32_v_f32m1
#define VSSEG4 __riscv_vsseg4e32_v_f32m1
#define VSSEG3 __riscv_vsseg3e32_v_f32m1
#define VSSEG2 __riscv_vsseg2e32_v_f32m1
#define VS __riscv_vse32_v_f32m1
#endif

int CNAME(BLASLONG N, BLASLONG M, FLOAT *A, BLASLONG lda, FLOAT *B) {
    VFLOAT_T col_0_v, col_1_v, col_2_v, col_3_v, col_4_v, col_5_v, col_6_v, col_7_v;
    BLASLONG m = 0;
    FLOAT *A_ptr;
    size_t vl;
    ptrdiff_t stride = lda * sizeof(FLOAT);
    for (; m + 8 <= M; m += 8) {
        for (BLASLONG n = 0; n < N; n += vl) {
            vl = VSETVL(N - n);
            VLSSEG8(&col_0_v, &col_1_v, &col_2_v, &col_3_v, &col_4_v, &col_5_v, &col_6_v, &col_7_v, A + lda * n + m, stride, vl);
            VSSEG8(B, col_0_v, col_1_v, col_2_v, col_3_v, col_4_v, col_5_v, col_6_v, col_7_v, vl);
            B += 8 * vl;
        }
    }
    switch (M % 8) {
        case 1:
            for (BLASLONG n = 0; n < N; n += vl) {
                vl = VSETVL(N - n);
                col_0_v = VLS(A + lda * n + m, stride, vl);
                VS(B, col_0_v, vl);
                B += vl;
            }
            break;
        case 2:
            for (BLASLONG n = 0; n < N; n += vl) {
                vl = VSETVL(N - n);
                VLSSEG2(&col_0_v, &col_1_v, A + lda * n + m, stride, vl);
                VSSEG2(B, col_0_v, col_1_v, vl);
                B += 2 * vl;
            }
            break;
        case 3:
            for (BLASLONG n = 0; n < N; n += vl) {
                vl = VSETVL(N - n);
                VLSSEG3(&col_0_v, &col_1_v, &col_2_v, A + lda * n + m, stride, vl);
                VSSEG3(B, col_0_v, col_1_v, col_2_v, vl);
                B += 3 * vl;
            }
            break;
        case 4:
            for (BLASLONG n = 0; n < N; n += vl) {
                vl = VSETVL(N - n);
                VLSSEG4(&col_0_v, &col_1_v, &col_2_v, &col_3_v, A + lda * n + m, stride, vl);
                VSSEG4(B, col_0_v, col_1_v, col_2_v, col_3_v, vl);
                B += 4 * vl;
            }
            break;
        case 5:
            for (BLASLONG n = 0; n < N; n += vl) {
                vl = VSETVL(N - n);
                VLSSEG5(&col_0_v, &col_1_v, &col_2_v, &col_3_v, &col_4_v, A + lda * n + m, stride, vl);
                VSSEG5(B, col_0_v, col_1_v, col_2_v, col_3_v, col_4_v, vl);
                B += 5 * vl;
            }
            break;
        case 6:
            for (BLASLONG n = 0; n < N; n += vl) {
                vl = VSETVL(N - n);
                VLSSEG6(&col_0_v, &col_1_v, &col_2_v, &col_3_v, &col_4_v, &col_5_v, A + lda * n + m, stride, vl);
                VSSEG6(B, col_0_v, col_1_v, col_2_v, col_3_v, col_4_v, col_5_v, vl);
                B += 6 * vl;
            }
            break;
        case 7:
            for (BLASLONG n = 0; n < N; n += vl) {
                vl = VSETVL(N - n);
                VLSSEG7(&col_0_v, &col_1_v, &col_2_v, &col_3_v, &col_4_v, &col_5_v, &col_6_v, A + lda * n + m, stride, vl);
                VSSEG7(B, col_0_v, col_1_v, col_2_v, col_3_v, col_4_v, col_5_v, col_6_v, vl);
                B += 7 * vl;
            }
            break;
    }
}
