// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

//
// Test LLVM IR codegen of shuffle instructions
//

__m256 x(__m256 a, __m256 b) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 8, i32 11, i32 7, i32 6, i32 12, i32 15>
  return _mm256_shuffle_ps(a, b, 203);
}

__m128d test_mm_permute_pd(__m128d a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 1, i32 0>
  return _mm_permute_pd(a, 1);
}

__m256d test_mm256_permute_pd(__m256d a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 1, i32 0, i32 3, i32 2>
  return _mm256_permute_pd(a, 5);
}

__m128 test_mm_permute_ps(__m128 a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 1, i32 0>
  return _mm_permute_ps(a, 0x1b);
}

__m256 test_mm256_permute_ps(__m256 a) {
  // Check if the mask is correct
  // CHECK: shufflevector{{.*}}<i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  return _mm256_permute_ps(a, 0x1b);
}
