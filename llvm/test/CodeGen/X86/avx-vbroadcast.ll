; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s
; XFAIL: *

; xfail this file for now because of PR8156, when it gets solved merge this with avx-splat.ll

; CHECK: vbroadcastsd (%
define <4 x i64> @A(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64* %ptr, align 8
  %vecinit.i = insertelement <4 x i64> undef, i64 %q, i32 0
  %vecinit2.i = insertelement <4 x i64> %vecinit.i, i64 %q, i32 1
  %vecinit4.i = insertelement <4 x i64> %vecinit2.i, i64 %q, i32 2
  %vecinit6.i = insertelement <4 x i64> %vecinit4.i, i64 %q, i32 3
  ret <4 x i64> %vecinit6.i
}

; CHECK: vbroadcastss (%
define <8 x i32> @B(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32* %ptr, align 4
  %vecinit.i = insertelement <8 x i32> undef, i32 %q, i32 0
  %vecinit2.i = insertelement <8 x i32> %vecinit.i, i32 %q, i32 1
  %vecinit4.i = insertelement <8 x i32> %vecinit2.i, i32 %q, i32 2
  %vecinit6.i = insertelement <8 x i32> %vecinit4.i, i32 %q, i32 3
  ret <8 x i32> %vecinit6.i
}

; CHECK: vbroadcastsd (%
define <4 x double> @C(double* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load double* %ptr, align 8
  %vecinit.i = insertelement <4 x double> undef, double %q, i32 0
  %vecinit2.i = insertelement <4 x double> %vecinit.i, double %q, i32 1
  %vecinit4.i = insertelement <4 x double> %vecinit2.i, double %q, i32 2
  %vecinit6.i = insertelement <4 x double> %vecinit4.i, double %q, i32 3
  ret <4 x double> %vecinit6.i
}

; CHECK: vbroadcastss (%
define <8 x float> @D(float* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load float* %ptr, align 4
  %vecinit.i = insertelement <8 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <8 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <8 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <8 x float> %vecinit4.i, float %q, i32 3
  ret <8 x float> %vecinit6.i
}

;;;; 128-bit versions

; CHECK: vbroadcastss (%
define <4 x float> @E(float* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load float* %ptr, align 4
  %vecinit.i = insertelement <4 x float> undef, float %q, i32 0
  %vecinit2.i = insertelement <4 x float> %vecinit.i, float %q, i32 1
  %vecinit4.i = insertelement <4 x float> %vecinit2.i, float %q, i32 2
  %vecinit6.i = insertelement <4 x float> %vecinit4.i, float %q, i32 3
  ret <4 x float> %vecinit6.i
}

; CHECK: vbroadcastss (%
define <4 x i32> @F(i32* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i32* %ptr, align 4
  %vecinit.i = insertelement <4 x i32> undef, i32 %q, i32 0
  %vecinit2.i = insertelement <4 x i32> %vecinit.i, i32 %q, i32 1
  %vecinit4.i = insertelement <4 x i32> %vecinit2.i, i32 %q, i32 2
  %vecinit6.i = insertelement <4 x i32> %vecinit4.i, i32 %q, i32 3
  ret <4 x i32> %vecinit6.i
}

; Unsupported vbroadcasts

; CHECK: _G
; CHECK-NOT: vbroadcastsd (%
; CHECK: ret
define <2 x i64> @G(i64* %ptr) nounwind uwtable readnone ssp {
entry:
  %q = load i64* %ptr, align 8
  %vecinit.i = insertelement <2 x i64> undef, i64 %q, i32 0
  %vecinit2.i = insertelement <2 x i64> %vecinit.i, i64 %q, i32 1
  ret <2 x i64> %vecinit2.i
}

; CHECK: _H
; CHECK-NOT: vbroadcastss
; CHECK: ret
define <4 x i32> @H(<4 x i32> %a) {
  %x = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  ret <4 x i32> %x
}

