; RUN: opt < %s -mem2reg -S | FileCheck %s

; This function is optnone, so the allocas should not be eliminated.

; CHECK-LABEL: @testfunc
; CHECK: alloca
; CHECK: alloca
define double @testfunc(i32 %i, double %j) optnone noinline {
	%I = alloca i32		; <i32*> [#uses=4]
	%J = alloca double		; <double*> [#uses=2]
	store i32 %i, i32* %I
	store double %j, double* %J
	%t1 = load i32, i32* %I		; <i32> [#uses=1]
	%t2 = add i32 %t1, 1		; <i32> [#uses=1]
	store i32 %t2, i32* %I
	%t3 = load i32, i32* %I		; <i32> [#uses=1]
	%t4 = sitofp i32 %t3 to double		; <double> [#uses=1]
	%t5 = load double, double* %J		; <double> [#uses=1]
	%t6 = fmul double %t4, %t5		; <double> [#uses=1]
	ret double %t6
}
