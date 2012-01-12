; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; PR11102
define <4 x float> @test1(<4 x float> %a) nounwind {
  %b = shufflevector <4 x float> zeroinitializer, <4 x float> %a, <4 x i32> <i32 2, i32 5, i32 undef, i32 undef>
  ret <4 x float> %b
; CHECK: test1:
; CHECK: vshufps
; CHECK: vpshufd
}

; rdar://10538417
define <3 x i64> @test2(<2 x i64> %v) nounwind readnone {
; CHECK: test2:
; CHECK: vxorpd
; CHECK: vperm2f128
  %1 = shufflevector <2 x i64> %v, <2 x i64> %v, <3 x i32> <i32 0, i32 1, i32 undef>
  %2 = shufflevector <3 x i64> zeroinitializer, <3 x i64> %1, <3 x i32> <i32 3, i32 4, i32 2>
  ret <3 x i64> %2
}

define <4 x i64> @test3(<4 x i64> %a, <4 x i64> %b) nounwind {
  %c = shufflevector <4 x i64> %a, <4 x i64> %b, <4 x i32> <i32 4, i32 5, i32 2, i32 undef>
  ret <4 x i64> %c
; CHECK: test3:
; CHECK: vperm2f128
}

define <8 x float> @test4(float %a) nounwind {
  %b = insertelement <8 x float> zeroinitializer, float %a, i32 0
  ret <8 x float> %b
; CHECK: test4:
; CHECK: vinsertf128
}

; rdar://10594409
define <8 x float> @test5(float* nocapture %f) nounwind uwtable readonly ssp {
entry:
  %0 = bitcast float* %f to <4 x float>*
  %1 = load <4 x float>* %0, align 16
; CHECK: vmovaps
; CHECK-NOT: vxorps
; CHECK-NOT: vinsertf128
  %shuffle.i = shufflevector <4 x float> %1, <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle.i
}

define <4 x double> @test6(double* nocapture %d) nounwind uwtable readonly ssp {
entry:
  %0 = bitcast double* %d to <2 x double>*
  %1 = load <2 x double>* %0, align 16
; CHECK: vmovaps
; CHECK-NOT: vxorps
; CHECK-NOT: vinsertf128
  %shuffle.i = shufflevector <2 x double> %1, <2 x double> <double 0.000000e+00, double undef>, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  ret <4 x double> %shuffle.i
}

define <16 x i16> @test7(<4 x i16> %a) nounwind {
; CHECK: test7

  %b = shufflevector <4 x i16> %a, <4 x i16> undef, <16 x i32> <i32 1, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i16> %b
}

