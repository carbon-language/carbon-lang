; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=-avx,+sse2                             | FileCheck %s --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=-avx,+sse2 -fast-isel -fast-isel-abort | FileCheck %s --check-prefix=SSE2
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=-avx2,+avx                             | FileCheck %s --check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=-avx2,+avx -fast-isel -fast-isel-abort | FileCheck %s --check-prefix=AVX

define float @test_sqrt_f32(float %a) {
; SSE2-LABEL: test_sqrt_f32
; SSE2:       sqrtss %xmm0, %xmm0
; AVX-LABEL:  test_sqrt_f32
; AVX:        vsqrtss %xmm0, %xmm0
  %res = call float @llvm.sqrt.f32(float %a)
  ret float %res
}
declare float @llvm.sqrt.f32(float) nounwind readnone

define double @test_sqrt_f64(double %a) {
; SSE2-LABEL: test_sqrt_f64
; SSE2:       sqrtsd %xmm0, %xmm0
; AVX-LABEL:  test_sqrt_f64
; AVX:        vsqrtsd %xmm0, %xmm0
  %res = call double @llvm.sqrt.f64(double %a)
  ret double %res
}
declare double @llvm.sqrt.f64(double) nounwind readnone


