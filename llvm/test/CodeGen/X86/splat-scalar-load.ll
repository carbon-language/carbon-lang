; RUN: llc < %s -mtriple=i386-apple-darwin -mattr=+sse2 | FileCheck %s
; rdar://7434544

define <2 x i64> @t1() nounwind ssp {
entry:
; CHECK: t1:
; CHECK: pshufd	$0, (%esp), %xmm0
  %array = alloca [8 x float], align 16
  %arrayidx = getelementptr inbounds [8 x float]* %array, i32 0, i32 0
  %tmp2 = load float* %arrayidx
  %vecinit = insertelement <4 x float> undef, float %tmp2, i32 0
  %vecinit5 = insertelement <4 x float> %vecinit, float %tmp2, i32 1
  %vecinit7 = insertelement <4 x float> %vecinit5, float %tmp2, i32 2
  %vecinit9 = insertelement <4 x float> %vecinit7, float %tmp2, i32 3
  %0 = bitcast <4 x float> %vecinit9 to <2 x i64>
  ret <2 x i64> %0
}

define <2 x i64> @t2() nounwind ssp {
entry:
; CHECK: t2:
; CHECK: pshufd	$85, (%esp), %xmm0
  %array = alloca [8 x float], align 4
  %arrayidx = getelementptr inbounds [8 x float]* %array, i32 0, i32 1
  %tmp2 = load float* %arrayidx
  %vecinit = insertelement <4 x float> undef, float %tmp2, i32 0
  %vecinit5 = insertelement <4 x float> %vecinit, float %tmp2, i32 1
  %vecinit7 = insertelement <4 x float> %vecinit5, float %tmp2, i32 2
  %vecinit9 = insertelement <4 x float> %vecinit7, float %tmp2, i32 3
  %0 = bitcast <4 x float> %vecinit9 to <2 x i64>
  ret <2 x i64> %0
}

define <4 x float> @t3(float %tmp1, float %tmp2, float %tmp3) nounwind readnone ssp {
entry:
; CHECK: t3:
; CHECK: pshufd	$-86, (%esp), %xmm0
  %0 = insertelement <4 x float> undef, float %tmp3, i32 0
  %1 = insertelement <4 x float> %0, float %tmp3, i32 1
  %2 = insertelement <4 x float> %1, float %tmp3, i32 2
  %3 = insertelement <4 x float> %2, float %tmp3, i32 3
  ret <4 x float> %3
}
