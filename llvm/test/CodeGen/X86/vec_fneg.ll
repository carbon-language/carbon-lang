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

; If we're bitcasting an integer to an FP vector, we should avoid the FPU/vector unit entirely.
; Make sure that we're flipping the sign bit and only the sign bit of each float.
; So instead of something like this:
;    movd	%rdi, %xmm0
;    xorps	.LCPI2_0(%rip), %xmm0
;
; We should generate:
;    movabsq     (put sign bit mask in integer register))
;    xorq        (flip sign bits)
;    movd        (move to xmm return register) 

define <2 x float> @fneg_bitcast(i64 %i) {
; CHECK-LABEL: fneg_bitcast:
; CHECK:	movabsq	$-9223372034707292160, %rax # imm = 0x8000000080000000
; CHECK-NEXT:	xorq	%rdi, %rax
; CHECK-NEXT:	movd	%rax, %xmm0
; CHECK-NEXT:	retq
  %bitcast = bitcast i64 %i to <2 x float>
  %fneg = fsub <2 x float> <float -0.0, float -0.0>, %bitcast
  ret <2 x float> %fneg
}
