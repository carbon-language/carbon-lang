; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=corei7 | FileCheck %s

; FNEG is defined as subtraction from -0.0.

; This test verifies that we use an xor with a constant to flip the sign bits; no subtraction needed.
define <4 x float> @t1(<4 x float> %Q) {
; CHECK-LABEL: t1:
; CHECK: xorps	{{.*}}LCPI0_0{{.*}}, %xmm0
; CHECK-NEXT: retq
        %tmp = fsub <4 x float> < float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00 >, %Q
	ret <4 x float> %tmp
}

; This test verifies that we generate an FP subtraction because "0.0 - x" is not an fneg.
define <4 x float> @t2(<4 x float> %Q) {
; CHECK-LABEL: t2:
; CHECK: xorps	%[[X:xmm[0-9]+]], %[[X]]
; CHECK-NEXT: subps	%xmm0, %[[X]]
; CHECK-NEXT: movaps	%[[X]], %xmm0
; CHECK-NEXT: retq
        %tmp = fsub <4 x float> zeroinitializer, %Q
	ret <4 x float> %tmp
}
