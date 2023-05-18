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


int CNAME(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc) {
    vfloat32m1_t A_v, C_col_0, C_col_1, C_col_2, C_col_3, C_col_4, C_col_5, C_col_6, C_col_7;
    FLOAT *A_ptr, *B_ptr, *C_ptr_l, *C_ptr_s;
    size_t vl;
    BLASLONG n = 0;
    for (; n + 8 <= N; n += 8) {
        for (BLASLONG m = 0; m < M; m += vl) {
            vl = __riscv_vsetvl_e32m1(M - m);
            C_ptr_l = C + n * ldc + m;
            C_ptr_s = C_ptr_l;
            C_col_0 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
            C_col_1 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
            C_col_2 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
            C_col_3 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
            C_col_4 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
            C_col_5 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
            C_col_6 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
            C_col_7 = __riscv_vle32_v_f32m1(C_ptr_l, vl);
            A_ptr = A + m * K;
            B_ptr = B + n * K;
            for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {
                A_v = __riscv_vle32_v_f32m1(A_ptr, vl);
                A_v = __riscv_vfmul_vf_f32m1(A_v, alpha, vl);
                C_col_0 = __riscv_vfmacc_vf_f32m1(C_col_0, *B_ptr, A_v, vl); ++B_ptr;
                C_col_1 = __riscv_vfmacc_vf_f32m1(C_col_1, *B_ptr, A_v, vl); ++B_ptr;
                C_col_2 = __riscv_vfmacc_vf_f32m1(C_col_2, *B_ptr, A_v, vl); ++B_ptr;
                C_col_3 = __riscv_vfmacc_vf_f32m1(C_col_3, *B_ptr, A_v, vl); ++B_ptr;
                C_col_4 = __riscv_vfmacc_vf_f32m1(C_col_4, *B_ptr, A_v, vl); ++B_ptr;
                C_col_5 = __riscv_vfmacc_vf_f32m1(C_col_5, *B_ptr, A_v, vl); ++B_ptr;
                C_col_6 = __riscv_vfmacc_vf_f32m1(C_col_6, *B_ptr, A_v, vl); ++B_ptr;
                C_col_7 = __riscv_vfmacc_vf_f32m1(C_col_7, *B_ptr, A_v, vl); ++B_ptr;
            }
            __riscv_vse32_v_f32m1(C_ptr_s, C_col_0, vl); C_ptr_s += ldc;
            __riscv_vse32_v_f32m1(C_ptr_s, C_col_1, vl); C_ptr_s += ldc;
            __riscv_vse32_v_f32m1(C_ptr_s, C_col_2, vl); C_ptr_s += ldc;
            __riscv_vse32_v_f32m1(C_ptr_s, C_col_3, vl); C_ptr_s += ldc;
            __riscv_vse32_v_f32m1(C_ptr_s, C_col_4, vl); C_ptr_s += ldc;
            __riscv_vse32_v_f32m1(C_ptr_s, C_col_5, vl); C_ptr_s += ldc;
            __riscv_vse32_v_f32m1(C_ptr_s, C_col_6, vl); C_ptr_s += ldc;
            __riscv_vse32_v_f32m1(C_ptr_s, C_col_7, vl);
        }
    }
    switch (N % 8) {
        case 1:
            for (BLASLONG m = 0; m < M; m += vl) {
                vl = __riscv_vsetvl_e32m1(M - m);
                C_ptr_l = C + n * ldc + m;
                C_ptr_s = C_ptr_l;
                C_col_0 = __riscv_vle32_v_f32m1(C_ptr_l, vl);
                A_ptr = A + m * K;
                B_ptr = B + n * K;
                for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {
                    A_v = __riscv_vle32_v_f32m1(A_ptr, vl);
                    A_v = __riscv_vfmul_vf_f32m1(A_v, alpha, vl);
                    C_col_0 = __riscv_vfmacc_vf_f32m1(C_col_0, *B_ptr, A_v, vl); ++B_ptr;
                }
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_0, vl);
            }
            break;
        case 2:
            for (BLASLONG m = 0; m < M; m += vl) {
                vl = __riscv_vsetvl_e32m1(M - m);
                C_ptr_l = C + n * ldc + m;
                C_ptr_s = C_ptr_l;
                C_col_0 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_1 = __riscv_vle32_v_f32m1(C_ptr_l, vl);
                A_ptr = A + m * K;
                B_ptr = B + n * K;
                for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {
                    A_v = __riscv_vle32_v_f32m1(A_ptr, vl);
                    A_v = __riscv_vfmul_vf_f32m1(A_v, alpha, vl);
                    C_col_0 = __riscv_vfmacc_vf_f32m1(C_col_0, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_1 = __riscv_vfmacc_vf_f32m1(C_col_1, *B_ptr, A_v, vl); ++B_ptr;
                }
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_0, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_1, vl);
            }
            break;
        case 3:
            for (BLASLONG m = 0; m < M; m += vl) {
                vl = __riscv_vsetvl_e32m1(M - m);
                C_ptr_l = C + n * ldc + m;
                C_ptr_s = C_ptr_l;
                C_col_0 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_1 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_2 = __riscv_vle32_v_f32m1(C_ptr_l, vl);
                A_ptr = A + m * K;
                B_ptr = B + n * K;
                for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {
                    A_v = __riscv_vle32_v_f32m1(A_ptr, vl);
                    A_v = __riscv_vfmul_vf_f32m1(A_v, alpha, vl);
                    C_col_0 = __riscv_vfmacc_vf_f32m1(C_col_0, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_1 = __riscv_vfmacc_vf_f32m1(C_col_1, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_2 = __riscv_vfmacc_vf_f32m1(C_col_2, *B_ptr, A_v, vl); ++B_ptr;
                }
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_0, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_1, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_2, vl);
            }
            break;
        case 4:
            for (BLASLONG m = 0; m < M; m += vl) {
                vl = __riscv_vsetvl_e32m1(M - m);
                C_ptr_l = C + n * ldc + m;
                C_ptr_s = C_ptr_l;
                C_col_0 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_1 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_2 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_3 = __riscv_vle32_v_f32m1(C_ptr_l, vl);
                A_ptr = A + m * K;
                B_ptr = B + n * K;
                for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {
                    A_v = __riscv_vle32_v_f32m1(A_ptr, vl);
                    A_v = __riscv_vfmul_vf_f32m1(A_v, alpha, vl);
                    C_col_0 = __riscv_vfmacc_vf_f32m1(C_col_0, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_1 = __riscv_vfmacc_vf_f32m1(C_col_1, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_2 = __riscv_vfmacc_vf_f32m1(C_col_2, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_3 = __riscv_vfmacc_vf_f32m1(C_col_3, *B_ptr, A_v, vl); ++B_ptr;
                }
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_0, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_1, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_2, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_3, vl);
            }
            break;
        case 5:
            for (BLASLONG m = 0; m < M; m += vl) {
                vl = __riscv_vsetvl_e32m1(M - m);
                C_ptr_l = C + n * ldc + m;
                C_ptr_s = C_ptr_l;
                C_col_0 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_1 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_2 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_3 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_4 = __riscv_vle32_v_f32m1(C_ptr_l, vl);
                A_ptr = A + m * K;
                B_ptr = B + n * K;
                for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {
                    A_v = __riscv_vle32_v_f32m1(A_ptr, vl);
                    A_v = __riscv_vfmul_vf_f32m1(A_v, alpha, vl);
                    C_col_0 = __riscv_vfmacc_vf_f32m1(C_col_0, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_1 = __riscv_vfmacc_vf_f32m1(C_col_1, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_2 = __riscv_vfmacc_vf_f32m1(C_col_2, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_3 = __riscv_vfmacc_vf_f32m1(C_col_3, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_4 = __riscv_vfmacc_vf_f32m1(C_col_4, *B_ptr, A_v, vl); ++B_ptr;
                }
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_0, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_1, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_2, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_3, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_4, vl);
            }
            break;
        case 6:
            for (BLASLONG m = 0; m < M; m += vl) {
                vl = __riscv_vsetvl_e32m1(M - m);
                C_ptr_l = C + n * ldc + m;
                C_ptr_s = C_ptr_l;
                C_col_0 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_1 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_2 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_3 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_4 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_5 = __riscv_vle32_v_f32m1(C_ptr_l, vl);
                A_ptr = A + m * K;
                B_ptr = B + n * K;
                for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {
                    A_v = __riscv_vle32_v_f32m1(A_ptr, vl);
                    A_v = __riscv_vfmul_vf_f32m1(A_v, alpha, vl);
                    C_col_0 = __riscv_vfmacc_vf_f32m1(C_col_0, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_1 = __riscv_vfmacc_vf_f32m1(C_col_1, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_2 = __riscv_vfmacc_vf_f32m1(C_col_2, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_3 = __riscv_vfmacc_vf_f32m1(C_col_3, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_4 = __riscv_vfmacc_vf_f32m1(C_col_4, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_5 = __riscv_vfmacc_vf_f32m1(C_col_5, *B_ptr, A_v, vl); ++B_ptr;
                }
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_0, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_1, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_2, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_3, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_4, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_5, vl);
            }
            break;
        case 7:
            for (BLASLONG m = 0; m < M; m += vl) {
                vl = __riscv_vsetvl_e32m1(M - m);
                C_ptr_l = C + n * ldc + m;
                C_ptr_s = C_ptr_l;
                C_col_0 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_1 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_2 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_3 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_4 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_5 = __riscv_vle32_v_f32m1(C_ptr_l, vl); C_ptr_l += ldc;
                C_col_6 = __riscv_vle32_v_f32m1(C_ptr_l, vl);
                A_ptr = A + m * K;
                B_ptr = B + n * K;
                for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {
                    A_v = __riscv_vle32_v_f32m1(A_ptr, vl);
                    A_v = __riscv_vfmul_vf_f32m1(A_v, alpha, vl);
                    C_col_0 = __riscv_vfmacc_vf_f32m1(C_col_0, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_1 = __riscv_vfmacc_vf_f32m1(C_col_1, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_2 = __riscv_vfmacc_vf_f32m1(C_col_2, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_3 = __riscv_vfmacc_vf_f32m1(C_col_3, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_4 = __riscv_vfmacc_vf_f32m1(C_col_4, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_5 = __riscv_vfmacc_vf_f32m1(C_col_5, *B_ptr, A_v, vl); ++B_ptr;
                    C_col_6 = __riscv_vfmacc_vf_f32m1(C_col_6, *B_ptr, A_v, vl); ++B_ptr;
                }
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_0, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_1, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_2, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_3, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_4, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_5, vl); C_ptr_s += ldc;
                __riscv_vse32_v_f32m1(C_ptr_s, C_col_6, vl);
            }
            break;
    }
    return 0;
}
