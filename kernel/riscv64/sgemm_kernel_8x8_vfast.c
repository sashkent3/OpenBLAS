// #include <assert.h>

#include "common.h"


int CNAME(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc) {
    // assume that SGEMM_UNROLL_M == VLMAX
    // assert(__riscv_vsetvlmax_e32m1() == SGEMM_UNROLL_M);
    // assert(SGEMM_UNROLL_M == 8);
    // assert(SGEMM_UNROLL_N == 8);
    // printf("SGEMM A:\n");
    // for (size_t i = 0; i < M * K; ++i) {
    //     printf("%g ", A[i]);
    // }
    // printf("\nSGEMM B:\n");
    // for (size_t i = 0; i < K * N; ++i) {
    //     printf("%g ", B[i]);
    // }
    // printf("\n");
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
