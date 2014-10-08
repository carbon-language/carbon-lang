// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -ffreestanding -target-feature +avx512bw -target-feature +avx512vl -emit-llvm -o - -Werror | FileCheck %s

#include <immintrin.h>

__mmask32 test_mm256_cmpeq_epi8_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.256
  return (__mmask32)_mm256_cmpeq_epi8_mask(__a, __b);
}

__mmask32 test_mm256_mask_cmpeq_epi8_mask(__mmask32 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.256
  return (__mmask32)_mm256_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm_cmpeq_epi8_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.128
  return (__mmask16)_mm_cmpeq_epi8_mask(__a, __b);
}

__mmask16 test_mm_mask_cmpeq_epi8_mask(__mmask16 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.128
  return (__mmask16)_mm_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask16 test_mm256_cmpeq_epi16_mask(__m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.256
  return (__mmask16)_mm256_cmpeq_epi16_mask(__a, __b);
}

__mmask16 test_mm256_mask_cmpeq_epi16_mask(__mmask16 __u, __m256i __a, __m256i __b) {
  // CHECK-LABEL: @test_mm256_mask_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.256
  return (__mmask16)_mm256_mask_cmpeq_epi16_mask(__u, __a, __b);
}

__mmask8 test_mm_cmpeq_epi16_mask(__m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.128
  return (__mmask8)_mm_cmpeq_epi16_mask(__a, __b);
}

__mmask8 test_mm_mask_cmpeq_epi16_mask(__mmask8 __u, __m128i __a, __m128i __b) {
  // CHECK-LABEL: @test_mm_mask_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.128
  return (__mmask8)_mm_mask_cmpeq_epi16_mask(__u, __a, __b);
}
