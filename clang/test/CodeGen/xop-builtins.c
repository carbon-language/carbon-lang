// RUN: %clang_cc1 %s -O3 -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - | FileCheck %s

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m128i test_mm_maccs_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacssww
  return _mm_maccs_epi16(a, b, c);
}

__m128i test_mm_macc_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsww
  return _mm_macc_epi16(a, b, c);
}

__m128i test_mm_maccsd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsswd
  return _mm_maccsd_epi16(a, b, c);
}

__m128i test_mm_maccd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacswd
  return _mm_maccd_epi16(a, b, c);
}

__m128i test_mm_maccs_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacssdd
  return _mm_maccs_epi32(a, b, c);
}

__m128i test_mm_macc_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsdd
  return _mm_macc_epi32(a, b, c);
}

__m128i test_mm_maccslo_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacssdql
  return _mm_maccslo_epi32(a, b, c);
}

__m128i test_mm_macclo_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsdql
  return _mm_macclo_epi32(a, b, c);
}

__m128i test_mm_maccshi_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacssdqh
  return _mm_maccshi_epi32(a, b, c);
}

__m128i test_mm_macchi_epi32(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmacsdqh
  return _mm_macchi_epi32(a, b, c);
}

__m128i test_mm_maddsd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmadcsswd
  return _mm_maddsd_epi16(a, b, c);
}

__m128i test_mm_maddd_epi16(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpmadcswd
  return _mm_maddd_epi16(a, b, c);
}
