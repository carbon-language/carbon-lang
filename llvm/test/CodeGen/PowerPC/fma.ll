; RUN: llc < %s -march=ppc32 -fp-contract=fast | FileCheck %s

define double @test_FMADD1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	ret double %E
; CHECK: test_FMADD1:
; CHECK: fmadd
; CHECK-NEXT: blr
}

define double @test_FMADD2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	ret double %E
; CHECK: test_FMADD2:
; CHECK: fmadd
; CHECK-NEXT: blr
}

define double @test_FMSUB(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %D, %C		; <double> [#uses=1]
	ret double %E
; CHECK: test_FMSUB:
; CHECK: fmsub
; CHECK-NEXT: blr
}

define double @test_FNMADD1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK: test_FNMADD1:
; CHECK: fnmadd
; CHECK-NEXT: blr
}

define double @test_FNMADD2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %C, %D		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK: test_FNMADD2:
; CHECK: fnmadd
; CHECK-NEXT: blr
}

define double @test_FNMSUB1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %C, %D		; <double> [#uses=1]
	ret double %E
; CHECK: test_FNMSUB1:
; CHECK: fnmsub
; CHECK-NEXT: blr
}

define double @test_FNMSUB2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %D, %C		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK: test_FNMSUB2:
; CHECK: fnmsub
; CHECK-NEXT: blr
}

define float @test_FNMSUBS(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fsub float %D, %C		; <float> [#uses=1]
	%F = fsub float -0.000000e+00, %E		; <float> [#uses=1]
	ret float %F
; CHECK: test_FNMSUBS:
; CHECK: fnmsubs
; CHECK-NEXT: blr
}
