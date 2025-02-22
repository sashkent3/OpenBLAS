SGEMM_UNROLL_M = 4 # should be no greater than VLMAX
SGEMM_UNROLL_N = 8
SEW = 64
LMUL = 1

TAB = ' ' * 4

def print_inner_body(N):
    print(TAB * 3 + 'B_ptr = B;')
    for i in range(N):
        print(TAB * 3 + 'C_ptr ', end='')
        if i == 0:
            print('= C + m; ', end='')
        else:
            print('+= ldc;  ', end='')
        print(f'col_{i} = __riscv_vle{SEW}_v_f{SEW}m{LMUL}(C_ptr, vl);')
    print(TAB * 3 + 'for (BLASLONG k = 0; k < K; ++k, A_ptr += vl) {')
    print(TAB * 4 + f'A_v = __riscv_vle{SEW}_v_f{SEW}m{LMUL}(A_ptr, vl);')
    print(TAB * 4 + f'A_v = __riscv_vfmul_vf_f{SEW}m{LMUL}(A_v, alpha, vl);')
    for i in range(N):
        print(TAB * 4 + f'col_{i} = __riscv_vfmacc_vf_f{SEW}m{LMUL}(col_{i}, *B_ptr, A_v, vl); ++B_ptr;')
    print(TAB * 3 + '}')
    for i in range(N):
        print(TAB * 3 + 'C_ptr ', end='')
        if i == 0:
            print('= C + m; ', end='')
        else:
            print('+= ldc;  ', end='')
        print(f'__riscv_vse{SEW}_v_f{SEW}m{LMUL}(C_ptr, col_{i}, vl);')

def print_outer_body(M, N):
    print(TAB * 2 + 'm = 0;')
    print(TAB * 2 + f'vl = __riscv_vsetvl_e{SEW}m{LMUL}({M});')
    print(TAB * 2 + 'A_ptr = A;')
    print(TAB * 2 + f'for (; m + {M} <= M; m += {M}) {{')
    print_inner_body(N)
    print(TAB * 2 + '}')
    for i in range((M - 1).bit_length() - 1, 0, -1):
        m = 1 << i
        print(TAB * 2 + f'if (M & {m}) {{')
        print(TAB * 3 + f'vl = __riscv_vsetvl_e{SEW}m{LMUL}({m});')
        print_inner_body(N)
        print(TAB * 3 + f'm += {m};')
        print(TAB * 2 + '}')
    print(TAB * 2 + 'if (M & 1) {')
    print(TAB * 3 + 'B_ptr = B;')
    print(TAB * 3 + ', '.join(f'acc_{i} = 0' for i in range(N)), end=';\n')
    print(TAB * 3 + 'for (BLASLONG k = 0; k < K; ++k, ++A_ptr) {')
    for i in range(N):
        print(TAB * 4 + f'acc_{i} += *A_ptr * *B_ptr; ++B_ptr;')
    print(TAB * 3 + '}')
    for i in range(N):
        print(TAB * 3 + 'C_ptr ', end='')
        if i == 0:
            print('= C + m; ', end='')
        else:
            print('+= ldc;  ', end='')
        print(f'*C_ptr += alpha * acc_{i};')
    print(TAB * 2 + '}')

print('''\
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
''')

print(f'''\
/***************************************************************************
THIS PROGRAM WAS GENERATED BY gemm_generator.py
WITH THE FOLLOWING PARAMETERS
SGEMM_UNROLL_M={SGEMM_UNROLL_M}
SGEMM_UNROLL_N={SGEMM_UNROLL_N}
SEW={SEW}
LMUL={LMUL}
*****************************************************************************/
''')

print('''\
#include "common.h"


int CNAME(BLASLONG M, BLASLONG N, BLASLONG K, FLOAT alpha, FLOAT* A, FLOAT* B, FLOAT* C, BLASLONG ldc) {\
''')
print(f'{TAB}vfloat{SEW}m{LMUL}_t A_v', *(f'col_{i}' for i in range(SGEMM_UNROLL_N)), sep=', ', end=';\n')
print(f'{TAB}FLOAT', ', '.join(f'acc_{i}' for i in range(SGEMM_UNROLL_N)), end=';\n')
print(f'''\
    FLOAT *A_ptr, *B_ptr, *C_ptr;
    size_t vl;
    BLASLONG n = 0, m;
    for (; n + {SGEMM_UNROLL_N} <= N; n += {SGEMM_UNROLL_N}, B += {SGEMM_UNROLL_N} * K, C += {SGEMM_UNROLL_N} * ldc) {{\
''')
print_outer_body(SGEMM_UNROLL_M, SGEMM_UNROLL_N)
print(TAB + '}')
for i in range((SGEMM_UNROLL_N - 1).bit_length() - 1, -1, -1):
    n = 1 << i
    print(TAB + f'if (N & {n}) {{')
    print_outer_body(SGEMM_UNROLL_M, n)
    if n > 1:
        print(TAB * 2 + f'n += {n};')
        print(TAB * 2 + f'B += {n} * K;')
        print(TAB * 2 + f'C += {n} * ldc;')
    print(TAB + '}')
print('}')