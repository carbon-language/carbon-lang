; RUN: llc < %s -mtriple=thumbv7-apple-ios -mattr=+vfp4 | FileCheck %s

define float @test_f32(float %a, float %b, float %c) nounwind readnone ssp {
entry:
; CHECK: test_f32
; CHECK: vfma.f32
  %call = tail call float @llvm.fma.f32(float %a, float %b, float %c) nounwind readnone
  ret float %call
}

define double @test_f64(double %a, double %b, double %c) nounwind readnone ssp {
entry:
; CHECK: test_f64
; CHECK: vfma.f64
  %call = tail call double @llvm.fma.f64(double %a, double %b, double %c) nounwind readnone
  ret double %call
}

define <2 x float> @test_v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind readnone ssp {
entry:
; CHECK: test_v2f32
; CHECK: vfma.f32
  %0 = tail call <2 x float> @llvm.fma.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind
  ret <2 x float> %0
}

declare float @llvm.fma.f32(float, float, float) nounwind readnone
declare double @llvm.fma.f64(double, double, double) nounwind readnone

declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
