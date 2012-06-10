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

__m128i test_mm_haddw_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddbw
  return _mm_haddw_epi8(a);
}

__m128i test_mm_haddd_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddbd
  return _mm_haddd_epi8(a);
}

__m128i test_mm_haddq_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddbq
  return _mm_haddq_epi8(a);
}

__m128i test_mm_haddd_epi16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddwd
  return _mm_haddd_epi16(a);
}

__m128i test_mm_haddq_epi16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddwq
  return _mm_haddq_epi16(a);
}

__m128i test_mm_haddq_epi32(__m128i a) {
  // CHECK: @llvm.x86.xop.vphadddq
  return _mm_haddq_epi32(a);
}

__m128i test_mm_haddw_epu8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddubw
  return _mm_haddw_epu8(a);
}

__m128i test_mm_haddd_epu8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddubd
  return _mm_haddd_epu8(a);
}

__m128i test_mm_haddq_epu8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddubq
  return _mm_haddq_epu8(a);
}

__m128i test_mm_haddd_epu16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphadduwd
  return _mm_haddd_epu16(a);
}

__m128i test_mm_haddq_epu16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphadduwq
  return _mm_haddq_epu16(a);
}

__m128i test_mm_haddq_epu32(__m128i a) {
  // CHECK: @llvm.x86.xop.vphaddudq
  return _mm_haddq_epu32(a);
}

__m128i test_mm_hsubw_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vphsubbw
  return _mm_hsubw_epi8(a);
}

__m128i test_mm_hsubd_epi16(__m128i a) {
  // CHECK: @llvm.x86.xop.vphsubwd
  return _mm_hsubd_epi16(a);
}

__m128i test_mm_hsubq_epi32(__m128i a) {
  // CHECK: @llvm.x86.xop.vphsubdq
  return _mm_hsubq_epi32(a);
}

__m128i test_mm_cmov_si128(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpcmov
  return _mm_cmov_si128(a, b, c);
}

__m256i test_mm256_cmov_si256(__m256i a, __m256i b, __m256i c) {
  // CHECK: @llvm.x86.xop.vpcmov.256
  return _mm256_cmov_si256(a, b, c);
}

__m128i test_mm_perm_epi8(__m128i a, __m128i b, __m128i c) {
  // CHECK: @llvm.x86.xop.vpperm
  return _mm_perm_epi8(a, b, c);
}

__m128i test_mm_rot_epi8(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vprotb
  return _mm_rot_epi8(a, b);
}

__m128i test_mm_rot_epi16(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vprotw
  return _mm_rot_epi16(a, b);
}

__m128i test_mm_rot_epi32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vprotd
  return _mm_rot_epi32(a, b);
}

__m128i test_mm_rot_epi64(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.xop.vprotq
  return _mm_rot_epi64(a, b);
}

__m128i test_mm_roti_epi8(__m128i a) {
  // CHECK: @llvm.x86.xop.vprotbi
  return _mm_roti_epi8(a, 1);
}

__m128i test_mm_roti_epi16(__m128i a) {
  // CHECK: @llvm.x86.xop.vprotwi
  return _mm_roti_epi16(a, 50);
}

__m128i test_mm_roti_epi32(__m128i a) {
  // CHECK: @llvm.x86.xop.vprotdi
  return _mm_roti_epi32(a, -30);
}

__m128i test_mm_roti_epi64(__m128i a) {
  // CHECK: @llvm.x86.xop.vprotqi
  return _mm_roti_epi64(a, 100);
}
