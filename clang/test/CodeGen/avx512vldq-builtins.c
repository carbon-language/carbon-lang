// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -ffreestanding -target-feature +avx512dq -target-feature +avx512vl -emit-llvm -o - -Werror | FileCheck %s

#include <immintrin.h>

__m256i test_mm256_mullo_epi64 (__m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mullo_epi64
  // CHECK: mul <4 x i64>
  return _mm256_mullo_epi64(__A, __B);
}

__m256i test_mm256_mask_mullo_epi64 (__m256i __W, __mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_mask_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.256
  return (__m256i) _mm256_mask_mullo_epi64 ( __W, __U, __A, __B);
}

__m256i test_mm256_maskz_mullo_epi64 (__mmask8 __U, __m256i __A, __m256i __B) {
  // CHECK-LABEL: @test_mm256_maskz_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.256
  return (__m256i) _mm256_maskz_mullo_epi64 (__U, __A, __B);
}

__m128i test_mm_mullo_epi64 (__m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mullo_epi64
  // CHECK: mul <2 x i64>
  return (__m128i) _mm_mullo_epi64(__A, __B);
}

__m128i test_mm_mask_mullo_epi64 (__m128i __W, __mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_mask_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.128
  return (__m128i) _mm_mask_mullo_epi64 ( __W, __U, __A, __B);
}

__m128i test_mm_maskz_mullo_epi64 (__mmask8 __U, __m128i __A, __m128i __B) {
  // CHECK-LABEL: @test_mm_maskz_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.128
  return (__m128i) _mm_maskz_mullo_epi64 (__U, __A, __B);
}

__m256d test_mm256_mask_andnot_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.256
  return (__m256d) _mm256_mask_andnot_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_andnot_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.256
  return (__m256d) _mm256_maskz_andnot_pd (__U, __A, __B);
}

__m128d test_mm_mask_andnot_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.128
  return (__m128d) _mm_mask_andnot_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_andnot_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.128
  return (__m128d) _mm_maskz_andnot_pd (__U, __A, __B);
}

__m256 test_mm256_mask_andnot_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.256
  return (__m256) _mm256_mask_andnot_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_andnot_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.256
  return (__m256) _mm256_maskz_andnot_ps (__U, __A, __B);
}

__m128 test_mm_mask_andnot_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.128
  return (__m128) _mm_mask_andnot_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_andnot_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.128
  return (__m128) _mm_maskz_andnot_ps (__U, __A, __B);
}

__m256d test_mm256_mask_and_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.256
  return (__m256d) _mm256_mask_and_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_and_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.256
  return (__m256d) _mm256_maskz_and_pd (__U, __A, __B);
}

__m128d test_mm_mask_and_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.128
  return (__m128d) _mm_mask_and_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_and_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.128
  return (__m128d) _mm_maskz_and_pd (__U, __A, __B);
}

__m256 test_mm256_mask_and_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.256
  return (__m256) _mm256_mask_and_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_and_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.256
  return (__m256) _mm256_maskz_and_ps (__U, __A, __B);
}

__m128 test_mm_mask_and_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.128
  return (__m128) _mm_mask_and_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_and_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.128
  return (__m128) _mm_maskz_and_ps (__U, __A, __B);
}

__m256d test_mm256_mask_xor_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.256
  return (__m256d) _mm256_mask_xor_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_xor_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.256
  return (__m256d) _mm256_maskz_xor_pd (__U, __A, __B);
}

__m128d test_mm_mask_xor_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.128
  return (__m128d) _mm_mask_xor_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_xor_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.128
  return (__m128d) _mm_maskz_xor_pd (__U, __A, __B);
}

__m256 test_mm256_mask_xor_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.256
  return (__m256) _mm256_mask_xor_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_xor_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.256
  return (__m256) _mm256_maskz_xor_ps (__U, __A, __B);
}

__m128 test_mm_mask_xor_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.128
    return (__m128) _mm_mask_xor_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_xor_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.128
  return (__m128) _mm_maskz_xor_ps (__U, __A, __B);
}

__m256d test_mm256_mask_or_pd (__m256d __W, __mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_mask_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.256
  return (__m256d) _mm256_mask_or_pd ( __W, __U, __A, __B);
}

__m256d test_mm256_maskz_or_pd (__mmask8 __U, __m256d __A, __m256d __B) {
  // CHECK-LABEL: @test_mm256_maskz_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.256
  return (__m256d) _mm256_maskz_or_pd (__U, __A, __B);
}

__m128d test_mm_mask_or_pd (__m128d __W, __mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_mask_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.128
  return (__m128d) _mm_mask_or_pd ( __W, __U, __A, __B);
}

__m128d test_mm_maskz_or_pd (__mmask8 __U, __m128d __A, __m128d __B) {
  // CHECK-LABEL: @test_mm_maskz_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.128
  return (__m128d) _mm_maskz_or_pd (__U, __A, __B);
}

__m256 test_mm256_mask_or_ps (__m256 __W, __mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_mask_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.256
  return (__m256) _mm256_mask_or_ps ( __W, __U, __A, __B);
}

__m256 test_mm256_maskz_or_ps (__mmask8 __U, __m256 __A, __m256 __B) {
  // CHECK-LABEL: @test_mm256_maskz_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.256
  return (__m256) _mm256_maskz_or_ps (__U, __A, __B);
}

__m128 test_mm_mask_or_ps (__m128 __W, __mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_mask_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.128
  return (__m128) _mm_mask_or_ps ( __W, __U, __A, __B);
}

__m128 test_mm_maskz_or_ps (__mmask8 __U, __m128 __A, __m128 __B) {
  // CHECK-LABEL: @test_mm_maskz_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.128
  return (__m128) _mm_maskz_or_ps(__U, __A, __B);
}
