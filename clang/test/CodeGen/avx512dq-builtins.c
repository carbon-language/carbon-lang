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

__m512i test_mm512_cvtpd_epi64(__m512d __A) {
  // CHECK-LABEL: @test_mm512_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd.512
  return _mm512_cvtpd_epi64(__A); 
}

__m512i test_mm512_mask_cvtpd_epi64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd.512
  return _mm512_mask_cvtpd_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtpd_epi64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtpd.512
  return _mm512_maskz_cvtpd_epi64(__U, __A); 
}

__m512i test_mm512_cvt_roundpd_epi64(__m512d __A) {
  // CHECK-LABEL: @test_mm512_cvt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_cvt_roundpd_epi64(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_mask_cvt_roundpd_epi64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_cvt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_mask_cvt_roundpd_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_maskz_cvt_roundpd_epi64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_maskz_cvt_roundpd_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_cvtpd_epu64(__m512d __A) {
  // CHECK-LABEL: @test_mm512_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd.512
  return _mm512_cvtpd_epu64(__A); 
}

__m512i test_mm512_mask_cvtpd_epu64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd.512
  return _mm512_mask_cvtpd_epu64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtpd_epu64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtpd.512
  return _mm512_maskz_cvtpd_epu64(__U, __A); 
}

__m512i test_mm512_cvt_roundpd_epu64(__m512d __A) {
  // CHECK-LABEL: @test_mm512_cvt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_cvt_roundpd_epu64(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_mask_cvt_roundpd_epu64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_cvt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_mask_cvt_roundpd_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_maskz_cvt_roundpd_epu64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_maskz_cvt_roundpd_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_cvtps_epi64(__m256 __A) {
  // CHECK-LABEL: @test_mm512_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps.512
  return _mm512_cvtps_epi64(__A); 
}

__m512i test_mm512_mask_cvtps_epi64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps.512
  return _mm512_mask_cvtps_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtps_epi64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtps.512
  return _mm512_maskz_cvtps_epi64(__U, __A); 
}

__m512i test_mm512_cvt_roundps_epi64(__m256 __A) {
  // CHECK-LABEL: @test_mm512_cvt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_cvt_roundps_epi64(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_mask_cvt_roundps_epi64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_mask_cvt_roundps_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_maskz_cvt_roundps_epi64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_maskz_cvt_roundps_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_cvtps_epu64(__m256 __A) {
  // CHECK-LABEL: @test_mm512_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps.512
  return _mm512_cvtps_epu64(__A); 
}

__m512i test_mm512_mask_cvtps_epu64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps.512
  return _mm512_mask_cvtps_epu64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvtps_epu64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtps.512
  return _mm512_maskz_cvtps_epu64(__U, __A); 
}

__m512i test_mm512_cvt_roundps_epu64(__m256 __A) {
  // CHECK-LABEL: @test_mm512_cvt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_cvt_roundps_epu64(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_mask_cvt_roundps_epu64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_mask_cvt_roundps_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_maskz_cvt_roundps_epu64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_maskz_cvt_roundps_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512d test_mm512_cvtepi64_pd(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2.pd.512
  return _mm512_cvtepi64_pd(__A); 
}

__m512d test_mm512_mask_cvtepi64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2.pd.512
  return _mm512_mask_cvtepi64_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_cvtepi64_pd(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtqq2.pd.512
  return _mm512_maskz_cvtepi64_pd(__U, __A); 
}

__m512d test_mm512_cvt_roundepi64_pd(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvt_roundepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_cvt_roundepi64_pd(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512d test_mm512_mask_cvt_roundepi64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvt_roundepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_mask_cvt_roundepi64_pd(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512d test_mm512_maskz_cvt_roundepi64_pd(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvt_roundepi64_pd
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_maskz_cvt_roundepi64_pd(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m256 test_mm512_cvtepi64_ps(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2.ps.512
  return _mm512_cvtepi64_ps(__A); 
}

__m256 test_mm512_mask_cvtepi64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2.ps.512
  return _mm512_mask_cvtepi64_ps(__W, __U, __A); 
}

__m256 test_mm512_maskz_cvtepi64_ps(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtqq2.ps.512
  return _mm512_maskz_cvtepi64_ps(__U, __A); 
}

__m256 test_mm512_cvt_roundepi64_ps(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvt_roundepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_cvt_roundepi64_ps(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m256 test_mm512_mask_cvt_roundepi64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvt_roundepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_mask_cvt_roundepi64_ps(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m256 test_mm512_maskz_cvt_roundepi64_ps(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvt_roundepi64_ps
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_maskz_cvt_roundepi64_ps(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_cvttpd_epi64(__m512d __A) {
  // CHECK-LABEL: @test_mm512_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd.512
  return _mm512_cvttpd_epi64(__A); 
}

__m512i test_mm512_mask_cvttpd_epi64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd.512
  return _mm512_mask_cvttpd_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvttpd_epi64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvttpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttpd.512
  return _mm512_maskz_cvttpd_epi64(__U, __A); 
}

__m512i test_mm512_cvtt_roundpd_epi64(__m512d __A) {
  // CHECK-LABEL: @test_mm512_cvtt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_cvtt_roundpd_epi64(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_mask_cvtt_roundpd_epi64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_mask_cvtt_roundpd_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_maskz_cvtt_roundpd_epi64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtt_roundpd_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_maskz_cvtt_roundpd_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_cvttpd_epu64(__m512d __A) {
  // CHECK-LABEL: @test_mm512_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd.512
  return _mm512_cvttpd_epu64(__A); 
}

__m512i test_mm512_mask_cvttpd_epu64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd.512
  return _mm512_mask_cvttpd_epu64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvttpd_epu64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvttpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttpd.512
  return _mm512_maskz_cvttpd_epu64(__U, __A); 
}

__m512i test_mm512_cvtt_roundpd_epu64(__m512d __A) {
  // CHECK-LABEL: @test_mm512_cvtt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_cvtt_roundpd_epu64(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_mask_cvtt_roundpd_epu64(__m512i __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_mask_cvtt_roundpd_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_maskz_cvtt_roundpd_epu64(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtt_roundpd_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_maskz_cvtt_roundpd_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_cvttps_epi64(__m256 __A) {
  // CHECK-LABEL: @test_mm512_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps.512
  return _mm512_cvttps_epi64(__A); 
}

__m512i test_mm512_mask_cvttps_epi64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps.512
  return _mm512_mask_cvttps_epi64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvttps_epi64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvttps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvttps.512
  return _mm512_maskz_cvttps_epi64(__U, __A); 
}

__m512i test_mm512_cvtt_roundps_epi64(__m256 __A) {
  // CHECK-LABEL: @test_mm512_cvtt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_cvtt_roundps_epi64(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_mask_cvtt_roundps_epi64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_mask_cvtt_roundps_epi64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_maskz_cvtt_roundps_epi64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtt_roundps_epi64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_maskz_cvtt_roundps_epi64(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_cvttps_epu64(__m256 __A) {
  // CHECK-LABEL: @test_mm512_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps.512
  return _mm512_cvttps_epu64(__A); 
}

__m512i test_mm512_mask_cvttps_epu64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps.512
  return _mm512_mask_cvttps_epu64(__W, __U, __A); 
}

__m512i test_mm512_maskz_cvttps_epu64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvttps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvttps.512
  return _mm512_maskz_cvttps_epu64(__U, __A); 
}

__m512i test_mm512_cvtt_roundps_epu64(__m256 __A) {
  // CHECK-LABEL: @test_mm512_cvtt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_cvtt_roundps_epu64(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_mask_cvtt_roundps_epu64(__m512i __W, __mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_mask_cvtt_roundps_epu64(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512i test_mm512_maskz_cvtt_roundps_epu64(__mmask8 __U, __m256 __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtt_roundps_epu64
  // CHECK: @llvm.x86.avx512.mask.cvtt.512
  return _mm512_maskz_cvtt_roundps_epu64(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512d test_mm512_cvtepu64_pd(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2.pd.512
  return _mm512_cvtepu64_pd(__A); 
}

__m512d test_mm512_mask_cvtepu64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2.pd.512
  return _mm512_mask_cvtepu64_pd(__W, __U, __A); 
}

__m512d test_mm512_maskz_cvtepu64_pd(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2.pd.512
  return _mm512_maskz_cvtepu64_pd(__U, __A); 
}

__m512d test_mm512_cvt_roundepu64_pd(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvt_roundepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_cvt_roundepu64_pd(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512d test_mm512_mask_cvt_roundepu64_pd(__m512d __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvt_roundepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_mask_cvt_roundepu64_pd(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512d test_mm512_maskz_cvt_roundepu64_pd(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvt_roundepu64_pd
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_maskz_cvt_roundepu64_pd(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m256 test_mm512_cvtepu64_ps(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2.ps.512
  return _mm512_cvtepu64_ps(__A); 
}

__m256 test_mm512_mask_cvtepu64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2.ps.512
  return _mm512_mask_cvtepu64_ps(__W, __U, __A); 
}

__m256 test_mm512_maskz_cvtepu64_ps(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvtepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvtuqq2.ps.512
  return _mm512_maskz_cvtepu64_ps(__U, __A); 
}

__m256 test_mm512_cvt_roundepu64_ps(__m512i __A) {
  // CHECK-LABEL: @test_mm512_cvt_roundepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_cvt_roundepu64_ps(__A, _MM_FROUND_TO_NEAREST_INT); 
}

__m256 test_mm512_mask_cvt_roundepu64_ps(__m256 __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_mask_cvt_roundepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_mask_cvt_roundepu64_ps(__W, __U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m256 test_mm512_maskz_cvt_roundepu64_ps(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: @test_mm512_maskz_cvt_roundepu64_ps
  // CHECK: @llvm.x86.avx512.mask.cvt.512
  return _mm512_maskz_cvt_roundepu64_ps(__U, __A, _MM_FROUND_TO_NEAREST_INT); 
}

__m512d test_mm512_range_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_range_pd(__A, __B, 4); 
}

__m512d test_mm512_mask_range_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_mask_range_pd(__W, __U, __A, __B, 4); 
}

__m512d test_mm512_maskz_range_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_range_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_maskz_range_pd(__U, __A, __B, 4); 
}

__m512d test_mm512_range_round_pd(__m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_range_round_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_range_round_pd(__A, __B, 4, 8); 
}

__m512d test_mm512_mask_range_round_pd(__m512d __W, __mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_mask_range_round_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_mask_range_round_pd(__W, __U, __A, __B, 4, 8); 
}

__m512d test_mm512_maskz_range_round_pd(__mmask8 __U, __m512d __A, __m512d __B) {
  // CHECK-LABEL: @test_mm512_maskz_range_round_pd
  // CHECK: @llvm.x86.avx512.mask.range.pd.512
  return _mm512_maskz_range_round_pd(__U, __A, __B, 4, 8); 
}

__m512 test_mm512_range_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_range_ps(__A, __B, 4); 
}

__m512 test_mm512_mask_range_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_mask_range_ps(__W, __U, __A, __B, 4); 
}

__m512 test_mm512_maskz_range_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_range_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_maskz_range_ps(__U, __A, __B, 4); 
}

__m512 test_mm512_range_round_ps(__m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_range_round_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_range_round_ps(__A, __B, 4, 8); 
}

__m512 test_mm512_mask_range_round_ps(__m512 __W, __mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_mask_range_round_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_mask_range_round_ps(__W, __U, __A, __B, 4, 8); 
}

__m512 test_mm512_maskz_range_round_ps(__mmask16 __U, __m512 __A, __m512 __B) {
  // CHECK-LABEL: @test_mm512_maskz_range_round_ps
  // CHECK: @llvm.x86.avx512.mask.range.ps.512
  return _mm512_maskz_range_round_ps(__U, __A, __B, 4, 8); 
}

__m512d test_mm512_reduce_pd(__m512d __A) {
  // CHECK-LABEL: @test_mm512_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_reduce_pd(__A, 4); 
}

__m512d test_mm512_mask_reduce_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_mask_reduce_pd(__W, __U, __A, 4); 
}

__m512d test_mm512_maskz_reduce_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_reduce_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_maskz_reduce_pd(__U, __A, 4); 
}

__m512 test_mm512_reduce_ps(__m512 __A) {
  // CHECK-LABEL: @test_mm512_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_reduce_ps(__A, 4); 
}

__m512 test_mm512_mask_reduce_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_mask_reduce_ps(__W, __U, __A, 4); 
}

__m512 test_mm512_maskz_reduce_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_reduce_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_maskz_reduce_ps(__U, __A, 4); 
}

__m512d test_mm512_reduce_round_pd(__m512d __A) {
  // CHECK-LABEL: @test_mm512_reduce_round_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_reduce_round_pd(__A, 4, 8); 
}

__m512d test_mm512_mask_reduce_round_pd(__m512d __W, __mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_mask_reduce_round_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_mask_reduce_round_pd(__W, __U, __A, 4, 8); 
}

__m512d test_mm512_maskz_reduce_round_pd(__mmask8 __U, __m512d __A) {
  // CHECK-LABEL: @test_mm512_maskz_reduce_round_pd
  // CHECK: @llvm.x86.avx512.mask.reduce.pd.512
  return _mm512_maskz_reduce_round_pd(__U, __A, 4, 8);
}

__m512 test_mm512_reduce_round_ps(__m512 __A) {
  // CHECK-LABEL: @test_mm512_reduce_round_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_reduce_round_ps(__A, 4, 8); 
}

__m512 test_mm512_mask_reduce_round_ps(__m512 __W, __mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_mask_reduce_round_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_mask_reduce_round_ps(__W, __U, __A, 4, 8); 
}

__m512 test_mm512_maskz_reduce_round_ps(__mmask16 __U, __m512 __A) {
  // CHECK-LABEL: @test_mm512_maskz_reduce_round_ps
  // CHECK: @llvm.x86.avx512.mask.reduce.ps.512
  return _mm512_maskz_reduce_round_ps(__U, __A, 4, 8); 
}

// REQUIRES: asserts
