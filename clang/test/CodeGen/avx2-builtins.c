// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m256 test_mm256_mpsadbw_epu8(__m256 x, __m256 y) {
  // CHECK: @llvm.x86.avx2.mpsadbw({{.*}}, {{.*}}, i32 3)
  return _mm256_mpsadbw_epu8(x, y, 3);
}

__m256 test_mm256_abs_epi8(__m256 a) {
  // CHECK: @llvm.x86.avx2.pabs.b
  return _mm256_abs_epi8(a);
}

__m256 test_mm256_abs_epi16(__m256 a) {
  // CHECK: @llvm.x86.avx2.pabs.w
  return _mm256_abs_epi16(a);
}

__m256 test_mm256_abs_epi32(__m256 a) {
  // CHECK: @llvm.x86.avx2.pabs.d
  return _mm256_abs_epi32(a);
}

__m256 test_mm256_packs_epi16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.packsswb
  return _mm256_packs_epi16(a, b);
}

__m256 test_mm256_packs_epi32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.packssdw
  return _mm256_packs_epi32(a, b);
}

__m256 test_mm256_packs_epu16(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.packuswb
  return _mm256_packus_epi16(a, b);
}

__m256 test_mm256_packs_epu32(__m256 a, __m256 b) {
  // CHECK: @llvm.x86.avx2.packusdw
  return _mm256_packus_epi32(a, b);
}

__m256 test_mm256_add_epi8(__m256 a, __m256 b) {
  // CHECK: add <32 x i8>
  return _mm256_add_epi8(a, b);
}

__m256 test_mm256_add_epi16(__m256 a, __m256 b) {
  // CHECK: add <16 x i16>
  return _mm256_add_epi16(a, b);
}

__m256 test_mm256_add_epi32(__m256 a, __m256 b) {
  // CHECK: add <8 x i32>
  return _mm256_add_epi32(a, b);
}

__m256 test_mm256_add_epi64(__m256 a, __m256 b) {
  // CHECK: add <4 x i64>
  return _mm256_add_epi64(a, b);
}

__m256 test_mm256_sub_epi8(__m256 a, __m256 b) {
  // CHECK: sub <32 x i8>
  return _mm256_sub_epi8(a, b);
}

__m256 test_mm256_sub_epi16(__m256 a, __m256 b) {
  // CHECK: sub <16 x i16>
  return _mm256_sub_epi16(a, b);
}

__m256 test_mm256_sub_epi32(__m256 a, __m256 b) {
  // CHECK: sub <8 x i32>
  return _mm256_sub_epi32(a, b);
}

__m256 test_mm256_sub_epi64(__m256 a, __m256 b) {
  // CHECK: sub <4 x i64>
  return _mm256_sub_epi64(a, b);
}
