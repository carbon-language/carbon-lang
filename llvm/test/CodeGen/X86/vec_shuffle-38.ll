; RUN: llc < %s -march=x86-64 | FileCheck %s

define <2 x double> @ld(<2 x double> %p) nounwind optsize ssp {
; CHECK: unpcklpd
  %shuffle = shufflevector <2 x double> %p, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %shuffle
}

define <2 x double> @hd(<2 x double> %p) nounwind optsize ssp {
; CHECK: unpckhpd
  %shuffle = shufflevector <2 x double> %p, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x double> %shuffle
}

define <2 x i64> @ldi(<2 x i64> %p) nounwind optsize ssp {
; CHECK: punpcklqdq
  %shuffle = shufflevector <2 x i64> %p, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %shuffle
}

define <2 x i64> @hdi(<2 x i64> %p) nounwind optsize ssp {
; CHECK: punpckhqdq
  %shuffle = shufflevector <2 x i64> %p, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %shuffle
}

; rdar://10050549
%struct.Float2 = type { float, float }

define <4 x float> @loadhpi(%struct.Float2* %vPtr, <4 x float> %vecin1) nounwind readonly ssp {
entry:
; CHECK: loadhpi
; CHECK-NOT: movq
; CHECK: movhps (
  %tmp1 = bitcast %struct.Float2* %vPtr to <1 x i64>*
  %addptr7 = getelementptr inbounds <1 x i64>* %tmp1, i64 0
  %tmp2 = bitcast <1 x i64>* %addptr7 to float*
  %tmp3 = load float* %tmp2, align 4
  %vec = insertelement <4 x float> undef, float %tmp3, i32 0
  %addptr.i12 = getelementptr inbounds float* %tmp2, i64 1
  %tmp4 = load float* %addptr.i12, align 4
  %vecin2 = insertelement <4 x float> %vec, float %tmp4, i32 1
  %shuffle = shufflevector <4 x float> %vecin1, <4 x float> %vecin2, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x float> %shuffle
}

; rdar://10119696
; CHECK: f
define <4 x float> @f(<4 x float> %x, double* nocapture %y) nounwind uwtable readonly ssp {
entry:
  ; CHECK: movsd  (%
  ; CHECK-NEXT: movsd  %xmm
  %u110.i = load double* %y, align 1
  %tmp8.i = insertelement <2 x double> undef, double %u110.i, i32 0
  %tmp9.i = bitcast <2 x double> %tmp8.i to <4 x float>
  %shuffle.i = shufflevector <4 x float> %x, <4 x float> %tmp9.i, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  ret <4 x float> %shuffle.i
}

