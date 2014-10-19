; RUN: llc < %s -march=ppc32 -fp-contract=fast -mattr=-vsx | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -fp-contract=fast -mattr=+vsx -mcpu=pwr7 | FileCheck -check-prefix=CHECK-VSX %s

declare double @dummy1(double) #0
declare double @dummy2(double, double) #0
declare double @dummy3(double, double, double) #0

define double @test_FMADD1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %C, %D		; <double> [#uses=1]
	ret double %E
; CHECK-LABEL: test_FMADD1:
; CHECK: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD1:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	ret double %E
; CHECK-LABEL: test_FMADD2:
; CHECK: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD2:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %D, %C		; <double> [#uses=1]
	ret double %E
; CHECK-LABEL: test_FMSUB1:
; CHECK: fmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB1:
; CHECK-VSX: xsmsubmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB2(double %A, double %B, double %C, double %D) {
	%E = fmul double %A, %B 	; <double> [#uses=2]
	%F = fadd double %E, %C 	; <double> [#uses=1]
	%G = fsub double %E, %D 	; <double> [#uses=1]
	%H = call double @dummy2(double %F, double %G)      ; <double> [#uses=1]
	ret double %H
; CHECK-LABEL: test_FMSUB2:
; CHECK: fmadd
; CHECK-NEXT: fmsub

; CHECK-VSX-LABEL: test_FMSUB2:
; CHECK-VSX: xsmaddadp
; CHECK-VSX-NEXT: xsmsubmdp
}

define double @test_FNMADD1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK-LABEL: test_FNMADD1:
; CHECK: fnmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMADD1:
; CHECK-VSX: xsnmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FNMADD2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %C, %D		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK-LABEL: test_FNMADD2:
; CHECK: fnmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMADD2:
; CHECK-VSX: xsnmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FNMSUB1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %C, %D		; <double> [#uses=1]
	ret double %E
; CHECK-LABEL: test_FNMSUB1:
; CHECK: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMSUB1:
; CHECK-VSX: xsnmsubmdp
}

define double @test_FNMSUB2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %D, %C		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK-LABEL: test_FNMSUB2:
; CHECK: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMSUB2:
; CHECK-VSX: xsnmsubmdp
; CHECK-VSX-NEXT: blr
}

define float @test_FNMSUBS(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fsub float %D, %C		; <float> [#uses=1]
	%F = fsub float -0.000000e+00, %E		; <float> [#uses=1]
	ret float %F
; CHECK-LABEL: test_FNMSUBS:
; CHECK: fnmsubs
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMSUBS:
; CHECK-VSX: fnmsubs
; CHECK-VSX-NEXT: blr
}
