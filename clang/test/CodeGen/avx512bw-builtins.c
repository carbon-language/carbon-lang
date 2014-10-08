// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -ffreestanding -target-feature +avx512bw -emit-llvm -o - -Werror | FileCheck %s

#include <immintrin.h>

__mmask64 test_mm512_cmpeq_epi8_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.512
  return (__mmask64)_mm512_cmpeq_epi8_mask(__a, __b);
}

__mmask64 test_mm512_mask_cmpeq_epi8_mask(__mmask64 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi8_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.b.512
  return (__mmask64)_mm512_mask_cmpeq_epi8_mask(__u, __a, __b);
}

__mmask32 test_mm512_cmpeq_epi16_mask(__m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.512
  return (__mmask32)_mm512_cmpeq_epi16_mask(__a, __b);
}

__mmask32 test_mm512_mask_cmpeq_epi16_mask(__mmask32 __u, __m512i __a, __m512i __b) {
  // CHECK-LABEL: @test_mm512_mask_cmpeq_epi16_mask
  // CHECK: @llvm.x86.avx512.mask.pcmpeq.w.512
  return (__mmask32)_mm512_mask_cmpeq_epi16_mask(__u, __a, __b);
}
