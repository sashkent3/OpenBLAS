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
#define VBOOL_T vbool8_t
#define VMXOR __riscv_vmxor_mm_b8
#define VMAND __riscv_vmand_mm_b8
#define VFLOAT_T vfloat64m8_t
#define VFLOAT_T_M1 vfloat64m1_t
#define VL __riscv_vle64_v_f64m8
#define VLS __riscv_vlse64_v_f64m8
#define VSETVL __riscv_vsetvl_e64m8
#define VFABS __riscv_vfabs_v_f64m8
#define VMFGT_V __riscv_vmfgt_vv_f64m8_b8
#define VMFGT_F __riscv_vmfgt_vf_f64m8_b8
#define VFDIV_V_M __riscv_vfdiv_vv_f64m8_m
#define VFDIV_F __riscv_vfdiv_vf_f64m8
#define VFMUL_F __riscv_vfmul_vf_f64m8
#define VFMUL_V __riscv_vfmul_vv_f64m8
#define VFMUL_V_M __riscv_vfmul_vv_f64m8_m
#define VFADD_F_M __riscv_vfadd_vf_f64m8_m
#define VFADD_V __riscv_vfadd_vv_f64m8
#define VFADD_V_M __riscv_vfadd_vv_f64m8_m
#define VMERGE_F __riscv_vmerge_vvm_f64m8
#define VFMV_S_F __riscv_vfmv_s_f_f64m1
#define VFMV_V_F __riscv_vfmv_v_f_f64m8
#define VFREDMAX __riscv_vfredmax_vs_f64m8_f64m1
#define VFREDSUM __riscv_vfredusum_vs_f64m8_f64m1
#define VFMV_F_S __riscv_vfmv_f_s_f64m1_f64
#define VFMACC __riscv_vfmacc_vv_f64m8
#define FABS fabs
#define SQRT sqrt
#else
#define VBOOL_T vbool4_t
#define VMXOR __riscv_vmxor_mm_b4
#define VMAND __riscv_vmand_mm_b4
#define VFLOAT_T vfloat32m8_t
#define VFLOAT_T_M1 vfloat32m1_t
#define VL __riscv_vle32_v_f32m8
#define VLS __riscv_vlse32_v_f32m8
#define VSETVL __riscv_vsetvl_e32m8
#define VFABS __riscv_vfabs_v_f32m8
#define VMFGT_V __riscv_vmfgt_vv_f32m8_b4
#define VMFGT_F __riscv_vmfgt_vf_f32m8_b4
#define VFDIV_V_M __riscv_vfdiv_vv_f32m8_m
#define VFDIV_F __riscv_vfdiv_vf_f32m8
#define VFMUL_F __riscv_vfmul_vf_f32m8
#define VFMUL_V __riscv_vfmul_vv_f32m8
#define VFMUL_V_M __riscv_vfmul_vv_f32m8_m
#define VFADD_F_M __riscv_vfadd_vf_f32m8_m
#define VFADD_V __riscv_vfadd_vv_f32m8
#define VFADD_V_M __riscv_vfadd_vv_f32m8_m
#define VMERGE_F __riscv_vmerge_vvm_f32m8
#define VFMV_S_F __riscv_vfmv_s_f_f32m1
#define VFMV_V_F __riscv_vfmv_v_f_f32m8
#define VFREDMAX __riscv_vfredmax_vs_f32m8_f32m1
#define VFREDSUM __riscv_vfredusum_vs_f32m8_f32m1
#define VFMV_F_S __riscv_vfmv_f_s_f32m1_f32
#define VFMACC __riscv_vfmacc_vv_f32m8
#define FABS fabsf
#define SQRT sqrtf
#endif

// #if defined(DOUBLE)
// #define VFMACC __riscv_vfmacc_vv_f64m8
// #else
// #define VFMACC __riscv_vfmacc_vv_f32m4
// #endif

// #if defined(DOUBLE)
// #define VFLOAT_T vfloat64m8_t
// #define VL __riscv_vle64_v_f64m8
// #define VLS __riscv_vlse64_v_f64m8
// #define VSETVL __riscv_vsetvl_e64m8
// #else
// #define VFLOAT_T vfloat32m4_t
// #define VL __riscv_vle32_v_f32m4
// #define VLS __riscv_vlse32_v_f32m4
// #define VSETVL __riscv_vsetvl_e32m4
// #endif

// #if defined(DOUBLE)
// #define VFLOAT_RES_T_M1 vfloat64m1_t
// #define VFLOAT_RES_T vfloat64m8_t
// #define VFREDSUM __riscv_vfredusum_vs_f64m8_f64m1
// #define VFMV_V_F __riscv_vfmv_v_f_f64m8
// #define VFMV_S_F __riscv_vfmv_s_f_f64m1
// #define VFMV_F_S __riscv_vfmv_f_s_f64m1_f64
// #define SQRT sqrt
// #else
// #define VFLOAT_RES_T_M1 vfloat32m1_t
// #define VFLOAT_RES_T vfloat32m4_t
// #define VFREDSUM __riscv_vfredusum_vs_f32m4_f32m1
// #define VFMV_V_F __riscv_vfmv_v_f_f32m4
// #define VFMACC __riscv_vfmacc_vv_f32m4
// #define VFMV_S_F __riscv_vfmv_s_f_f32m1
// #define VFMV_F_S __riscv_vfmv_f_s_f32m1_f32
// #define SQRT sqrtf
// #endif

// #ifdef DOUBLE
// #define CNAME_CORR dnrm2_corr
// #define CNAME_MINE dnrm2_mine
// #else
// #define CNAME_CORR snrm2_corr
// #define CNAME_MINE snrm2_mine
// #endif

// FLOAT CNAME_CORR(BLASLONG n, FLOAT *x, BLASLONG inc_x) {
//     BLASLONG i=0;
//     FLOAT scale = 0.0;
//     FLOAT ssq   = 1.0;
//     FLOAT absxi = 0.0;

//     if (n <= 0 || inc_x <= 0) return(0.0);
//     if ( n == 1 ) return( FABS(x[0]) );

//     n *= inc_x;
//     while(i < n) {

//         if ( x[i] != 0.0 )
//         {
//             absxi = FABS( x[i] );
//             if ( scale < absxi )
//             {
//                 ssq = 1 + ssq * ( scale / absxi ) * ( scale / absxi );
//                 scale = absxi ;
//             }
//             else
//             {
//                 ssq += ( absxi/scale ) * ( absxi/scale );
//             }

//         }
//         i += inc_x;
//     }
//     scale = scale * SQRT( ssq );
//     return(scale);
// }

FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x) {
//     if (n <= 0) return 0;
//     size_t vl_start = VSETVL(n);
//     VFLOAT_RES_T res_chunks_v = VFMV_V_F(0, vl_start);
//     size_t vl = vl_start;
//     if (inc_x == 1) {
//         VFLOAT_T x_v = VL(x, vl);
//         res_chunks_v = VFMACC(res_chunks_v, x_v, x_v, vl);
//         for (BLASLONG offset = vl_start; offset < n; offset += vl) {
//             vl = VSETVL(n - offset);
//             x_v = VL(x + offset, vl);
//             res_chunks_v = VFMACC(res_chunks_v, x_v, x_v, vl);
//         }
//     } else {
//         ptrdiff_t stride_x = inc_x * sizeof(FLOAT);
//         VFLOAT_T x_v = VLS(x, stride_x, vl);
//         res_chunks_v = VFMACC(res_chunks_v, x_v, x_v, vl);
//         for (BLASLONG offset = vl_start; offset < n; offset += vl) {
//             vl = VSETVL(n - offset);
//             x_v = VLS(x + offset * inc_x, stride_x, vl);
//             res_chunks_v = VFMACC(res_chunks_v, x_v, x_v, vl);
//         }
//     }
//     VFLOAT_RES_T_M1 zero_v = VFMV_S_F(0, vl_start);
//     VFLOAT_RES_T_M1 res_v = VFREDSUM(res_chunks_v, zero_v, vl_start);
//     return SQRT(VFMV_F_S(res_v));
// }

//     if (n <= 0) return 0.0;

//     FLOAT scale_first = FABS(*x);
//     if (n == 1) return scale_first;
//     if (inc_x == 0) return SQRT((FLOAT) n) * scale_first;
//     size_t vl_start = VSETVL(n - 1), vl;
//     VFLOAT_T ssq_v = VFMV_V_F(1.0, vl_start), scale_v, scale_coef_v, x_v;
//     VBOOL_T mask, mask_rescale, mask_nonzero;
//     if (inc_x == 1) {
//         scale_v = VL(x + 1, vl_start);
//         scale_v = VFABS(scale_v, vl_start);
//         for (size_t offset = vl_start + 1   ; offset < n; offset += vl) {
//             vl = VSETVL(n - offset);
//             x_v = VL(x + offset, vl);
//             x_v = VFABS(x_v, vl);
//             mask_rescale = VMFGT_V(x_v, scale_v, vl);
//             mask_nonzero = VMFGT_F(scale_v, 0.0, vl);

//             mask = VMAND(mask_rescale, mask_nonzero, vl);
//             scale_coef_v = VFDIV_V_M(mask, scale_v, x_v, vl);
//             scale_coef_v = VFMUL_V_M(mask, scale_coef_v, scale_coef_v, vl);
//             ssq_v = VFMUL_V_M(mask, ssq_v, scale_coef_v, vl);
//             ssq_v = VFADD_F_M(mask, ssq_v, 1.0, vl);

//             // mask = VMAND(VMNOT(mask_rescale, vl), mask_nonzero, vl);
//             mask = VMXOR(mask, mask_nonzero, vl);
//             x_v = VFDIV_V_M(mask, x_v, scale_v, vl);
//             x_v = VFMUL_V_M(mask, x_v, x_v, vl);
//             ssq_v = VFADD_V_M(mask, ssq_v, x_v, vl);

//             scale_v = VMERGE_F(x_v, scale_v, mask, vl);
//         }
//     } else {
//         ptrdiff_t stride_x = inc_x * sizeof(FLOAT);
//         scale_v = VLS(x, stride_x, vl_start);
//         scale_v = VFABS(scale_v, vl_start);
//         for (size_t offset = vl_start; offset < n; offset += vl) {
//             vl = VSETVL(n - offset);
//             x_v = VLS(x + offset * inc_x, stride_x, vl);
//             x_v = VFABS(x_v, vl);
//             mask_rescale = VMFGT_V(x_v, scale_v, vl);
//             mask_nonzero = VMFGT_F(scale_v, 0.0, vl);

//             mask = VMAND(mask_rescale, mask_nonzero, vl);
//             scale_coef_v = VFDIV_V_M(mask, scale_v, x_v, vl);
//             scale_coef_v = VFMUL_V_M(mask, scale_coef_v, scale_coef_v, vl);
//             ssq_v = VFMUL_V_M(mask, ssq_v, scale_coef_v, vl);
//             ssq_v = VFADD_F_M(mask, ssq_v, 1.0, vl);

//             // mask = VMAND(VMNOT(mask_rescale, vl), mask_nonzero, vl);
//             mask = VMXOR(mask, mask_nonzero, vl);
//             x_v = VFDIV_V_M(mask, x_v, scale_v, vl);
//             x_v = VFMUL_V_M(mask, x_v, x_v, vl);
//             ssq_v = VFADD_V_M(mask, ssq_v, x_v, vl);

//             scale_v = VMERGE_F(x_v, scale_v, mask, vl);
//         }
//     }

//     FLOAT scale_max = VFMV_F_S(VFREDMAX(scale_v, VFMV_S_F(scale_first, vl_start), vl_start));
//     if (scale_max == 0) return 0.0;
//     scale_coef_v = VFDIV_F(scale_v, scale_max, vl_start);
//     scale_coef_v = VFMUL_V(scale_coef_v, scale_coef_v, vl_start);
//     ssq_v = VFMUL_V(ssq_v, scale_coef_v, vl_start);
//     scale_first = scale_first / scale_max;
//     scale_first *= scale_first;
//     VFLOAT_T_M1 res_v = VFREDSUM(ssq_v, VFMV_S_F(scale_first, vl_start), vl_start);
//     return scale_max * SQRT(VFMV_F_S(res_v));
// }
    if (n <= 0) return 0.0;
    FLOAT first = FABS(*x);
    if (n == 1) return first;
    if (inc_x == 0) return SQRT((FLOAT) n) * first;
    FLOAT scale = first, scale_new, rescale_coef;
    VFLOAT_T_M1 scale_v = VFMV_S_F(scale, 1);
    size_t vl_start = VSETVL(n - 1);
    VFLOAT_T ssq_v = VFMV_V_F(0.0, vl_start), x_v;
    size_t vl;
    if (inc_x == 1) {
        for (size_t offset = 1; offset < n; offset += vl) {
            vl = VSETVL(n - offset);
            x_v = VL(x + offset, vl);
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
        for (size_t offset = 1; offset < n; offset += vl) {
            vl = VSETVL(n - offset);
            x_v = VLS(x + offset * inc_x, stride_x, vl);
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

// FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x) {
//     FLOAT corr = CNAME_CORR(n, x, inc_x);
//     FLOAT mine = CNAME_MINE(n, x, inc_x);
//     if (FABS(corr - mine) > 1e-10) {
//         printf("\nn=%ld\n", n);
//         printf("corr=%g\n", corr);
//         printf("mine=%g\n", mine);
//         if (n <= 50) {
//             if (n > 0) printf("%g", x[0]);
//             for (size_t i = 1; i < n; ++i) {
//                 printf(", %g", x[inc_x * i]);
//             }
//             printf("\n");
//         }
//     }
//     return corr;
// }

// ssq_old = (x0 / scale_old) ^ 2 + ... + (xk / scale_old) ^ 2 = (x0 ^ 2 + ... + xk ^ 2) / scale_old ^ 2
// ssq = (x0 / scale) ^ 2 + ... + (xk / scale) ^ 2 = ssq_old * (scale_old / scale_new) ^ 2

// FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x) {

