// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=i686-apple-darwin -target-feature +ssse3 -O1 -S -o - | FileCheck %s

#define _mm_alignr_epi8(a, b, n) (__builtin_ia32_palignr128((a), (b), (n)))
typedef __attribute__((vector_size(16))) int int4;

// CHECK: palignr $15, %xmm1, %xmm0
int4 align1(int4 a, int4 b) { return _mm_alignr_epi8(a, b, 15); }
// CHECK: ret
// CHECK: ret
// CHECK-NOT: palignr
int4 align2(int4 a, int4 b) { return _mm_alignr_epi8(a, b, 16); }
// CHECK: psrldq $1, %xmm0
int4 align3(int4 a, int4 b) { return _mm_alignr_epi8(a, b, 17); }
// CHECK: xor
int4 align4(int4 a, int4 b) { return _mm_alignr_epi8(a, b, 32); }

#define _mm_alignr_pi8(a, b, n) (__builtin_ia32_palignr((a), (b), (n)))
typedef __attribute__((vector_size(8))) int int2;

// CHECK: palignr
int2 align5(int2 a, int2 b) { return _mm_alignr_pi8(a, b, 8); }

// CHECK: palignr
int2 align6(int2 a, int2 b) { return _mm_alignr_pi8(a, b, 9); }

// CHECK: palignr
int2 align7(int2 a, int2 b) { return _mm_alignr_pi8(a, b, 16); }

// CHECK: palignr
int2 align8(int2 a, int2 b) { return _mm_alignr_pi8(a, b, 7); }
