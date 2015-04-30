// RUN: %clang_cc1 %s -O0 -triple=x86_64-apple-darwin -ffreestanding -target-feature +avx512dq -emit-llvm -o - -Werror | FileCheck %s

#include <immintrin.h>
__m512i test_mm512_mullo_epi64 (__m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mullo_epi64
  // CHECK: mul <8 x i64>
  return (__m512i) ((__v8di) __A * (__v8di) __B);
}

__m512i test_mm512_mask_mullo_epi64 (__m512i __W, __mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_mask_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.512
  return (__m512i) _mm512_mask_mullo_epi64(__W, __U, __A, __B);
}

__m512i test_mm512_maskz_mullo_epi64 (__mmask8 __U, __m512i __A, __m512i __B) {
  // CHECK-LABEL: @test_mm512_maskz_mullo_epi64
  // CHECK: @llvm.x86.avx512.mask.pmull.q.512
  return (__m512i) _mm512_maskz_mullo_epi64(__U, __A, __B);
}

__m512d test_mm512_xor_pd (__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_xor_pd
  // CHECK: xor <8 x i64>
  return (__m512d) _mm512_xor_pd(__A, __B);
}

__m512d test_mm512_mask_xor_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.512
  return (__m512d) _mm512_mask_xor_pd(__W, __U, __A, __B);
}

__m512d test_mm512_maskz_xor_pd (__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_xor_pd
  // CHECK: @llvm.x86.avx512.mask.xor.pd.512
  return (__m512d) _mm512_maskz_xor_pd(__U, __A, __B);
}

__m512 test_mm512_xor_ps (__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_xor_ps
  // CHECK: xor <16 x i32>
  return (__m512) _mm512_xor_ps(__A, __B);
}

__m512 test_mm512_mask_xor_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.512
  return (__m512) _mm512_mask_xor_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_xor_ps (__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_xor_ps
  // CHECK: @llvm.x86.avx512.mask.xor.ps.512
  return (__m512) _mm512_maskz_xor_ps(__U, __A, __B);
}

__m512d test_mm512_or_pd (__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_or_pd
  // CHECK: or <8 x i64>
  return (__m512d) _mm512_or_pd(__A, __B);
}

__m512d test_mm512_mask_or_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.512
  return (__m512d) _mm512_mask_or_pd(__W, __U, __A, __B);
}

__m512d test_mm512_maskz_or_pd (__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_or_pd
  // CHECK: @llvm.x86.avx512.mask.or.pd.512
  return (__m512d) _mm512_maskz_or_pd(__U, __A, __B);
}

__m512 test_mm512_or_ps (__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_or_ps
  // CHECK: or <16 x i32>
  return (__m512) _mm512_or_ps(__A, __B);
}

__m512 test_mm512_mask_or_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.512
  return (__m512) _mm512_mask_or_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_or_ps (__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_or_ps
  // CHECK: @llvm.x86.avx512.mask.or.ps.512
  return (__m512) _mm512_maskz_or_ps(__U, __A, __B);
}

__m512d test_mm512_and_pd (__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_and_pd
  // CHECK: and <8 x i64>
  return (__m512d) _mm512_and_pd(__A, __B);
}

__m512d test_mm512_mask_and_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.512
  return (__m512d) _mm512_mask_and_pd(__W, __U, __A, __B);
}

__m512d test_mm512_maskz_and_pd (__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_and_pd
  // CHECK: @llvm.x86.avx512.mask.and.pd.512
  return (__m512d) _mm512_maskz_and_pd(__U, __A, __B);
}

__m512 test_mm512_and_ps (__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_and_ps
  // CHECK: and <16 x i32>
  return (__m512) _mm512_and_ps(__A, __B);
}

__m512 test_mm512_mask_and_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.512
  return (__m512) _mm512_mask_and_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_and_ps (__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_and_ps
  // CHECK: @llvm.x86.avx512.mask.and.ps.512
  return (__m512) _mm512_maskz_and_ps(__U, __A, __B);
}

__m512d test_mm512_andnot_pd (__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.512
  return (__m512d) _mm512_andnot_pd(__A, __B);
}

__m512d test_mm512_mask_andnot_pd (__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.512
  return (__m512d) _mm512_mask_andnot_pd(__W, __U, __A, __B);
}

__m512d test_mm512_maskz_andnot_pd (__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_andnot_pd
  // CHECK: @llvm.x86.avx512.mask.andn.pd.512
  return (__m512d) _mm512_maskz_andnot_pd(__U, __A, __B);
}

__m512 test_mm512_andnot_ps (__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.512
  return (__m512) _mm512_andnot_ps(__A, __B);
}

__m512 test_mm512_mask_andnot_ps (__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.512
  return (__m512) _mm512_mask_andnot_ps(__W, __U, __A, __B);
}

__m512 test_mm512_maskz_andnot_ps (__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_andnot_ps
  // CHECK: @llvm.x86.avx512.mask.andn.ps.512
  return (__m512) _mm512_maskz_andnot_ps(__U, __A, __B);
}
