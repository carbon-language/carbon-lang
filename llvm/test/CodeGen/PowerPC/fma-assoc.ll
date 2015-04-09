; RUN: llc < %s -march=ppc32 -fp-contract=fast -mattr=-vsx | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -fp-contract=fast -mattr=+vsx -mcpu=pwr7 | FileCheck -check-prefix=CHECK-VSX %s

define double @test_FMADD_ASSOC1(double %A, double %B, double %C,
                                 double %D, double %E) {
	%F = fmul double %A, %B         ; <double> [#uses=1]
	%G = fmul double %C, %D         ; <double> [#uses=1]
	%H = fadd double %F, %G         ; <double> [#uses=1]
	%I = fadd double %H, %E         ; <double> [#uses=1]
	ret double %I
; CHECK-LABEL: test_FMADD_ASSOC1:
; CHECK: fmadd
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD_ASSOC1:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD_ASSOC2(double %A, double %B, double %C,
                                 double %D, double %E) {
	%F = fmul double %A, %B         ; <double> [#uses=1]
	%G = fmul double %C, %D         ; <double> [#uses=1]
	%H = fadd double %F, %G         ; <double> [#uses=1]
	%I = fadd double %E, %H         ; <double> [#uses=1]
	ret double %I
; CHECK-LABEL: test_FMADD_ASSOC2:
; CHECK: fmadd
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD_ASSOC2:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC1(double %A, double %B, double %C,
                                 double %D, double %E) {
	%F = fmul double %A, %B         ; <double> [#uses=1]
	%G = fmul double %C, %D         ; <double> [#uses=1]
	%H = fadd double %F, %G         ; <double> [#uses=1]
	%I = fsub double %H, %E         ; <double> [#uses=1]
	ret double %I
; CHECK-LABEL: test_FMSUB_ASSOC1:
; CHECK: fmsub
; CHECK-NEXT: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_ASSOC1:
; CHECK-VSX: xsmsubmdp
; CHECK-VSX-NEXT: xsmaddadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB_ASSOC2(double %A, double %B, double %C,
                                 double %D, double %E) {
	%F = fmul double %A, %B         ; <double> [#uses=1]
	%G = fmul double %C, %D         ; <double> [#uses=1]
	%H = fadd double %F, %G         ; <double> [#uses=1]
	%I = fsub double %E, %H         ; <double> [#uses=1]
	ret double %I
; CHECK-LABEL: test_FMSUB_ASSOC2:
; CHECK: fnmsub
; CHECK-NEXT: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB_ASSOC2:
; CHECK-VSX: xsnmsubmdp
; CHECK-VSX-NEXT: xsnmsubadp
; CHECK-VSX-NEXT: fmr
; CHECK-VSX-NEXT: blr
}

