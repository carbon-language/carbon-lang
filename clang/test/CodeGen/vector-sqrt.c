// RUN: %clang %s -march=haswell -O3 -S -o - | FileCheck %s

#include <x86intrin.h>

// CHECK-LABEL: sqrtd2
// CHECK:       vsqrtsd (%rdi), %xmm0, %xmm0
// CHECK-NEXT:  vsqrtsd 8(%rdi), %xmm1, %xmm1
// CHECK-NEXT:  vunpcklpd %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[0]
// CHECK-NEXT:  retq

__m128d sqrtd2(double* v) {
  return _mm_setr_pd(__builtin_sqrt(v[0]), __builtin_sqrt(v[1]));
}

// CHECK-LABEL: sqrtf4
// CHECK:       vsqrtss (%rdi), %xmm0, %xmm0
// CHECK-NEXT:  vsqrtss 4(%rdi), %xmm1, %xmm1
// CHECK-NEXT:  vsqrtss 8(%rdi), %xmm2, %xmm2
// CHECK-NEXT:  vsqrtss 12(%rdi), %xmm3, %xmm3
// CHECK-NEXT:  vinsertps $16, %xmm1, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm1[0],xmm0[2,3]
// CHECK-NEXT:  vinsertps $32, %xmm2, %xmm0, %xmm0 # xmm0 = xmm0[0,1],xmm2[0],xmm0[3]
// CHECK-NEXT:  vinsertps $48, %xmm3, %xmm0, %xmm0 # xmm0 = xmm0[0,1,2],xmm3[0]
// CHECK-NEXT:  retq

__m128 sqrtf4(float* v) {
  return _mm_setr_ps(__builtin_sqrtf(v[0]), __builtin_sqrtf(v[1]), __builtin_sqrtf(v[2]), __builtin_sqrtf(v[3]));
}
