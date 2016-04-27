// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Werror | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m128i test_mm_broadcastmb_epi64(__mmask8 __A) {
  // CHECK-LABEL: @test_mm_broadcastmb_epi64
  // CHECK: @llvm.x86.avx512.broadcastmb.128
  return _mm_broadcastmb_epi64(__A); 
}

__m256i test_mm256_broadcastmb_epi64(__mmask8 __A) {
  // CHECK-LABEL: @test_mm256_broadcastmb_epi64
  // CHECK: @llvm.x86.avx512.broadcastmb.256
  return _mm256_broadcastmb_epi64(__A); 
}

__m128i test_mm_broadcastmw_epi32(__mmask16 __A) {
  // CHECK-LABEL: @test_mm_broadcastmw_epi32
  // CHECK: @llvm.x86.avx512.broadcastmw.128
  return _mm_broadcastmw_epi32(__A); 
}

__m256i test_mm256_broadcastmw_epi32(__mmask16 __A) {
  // CHECK-LABEL: @test_mm256_broadcastmw_epi32
  // CHECK: @llvm.x86.avx512.broadcastmw.256
  return _mm256_broadcastmw_epi32(__A); 
}
