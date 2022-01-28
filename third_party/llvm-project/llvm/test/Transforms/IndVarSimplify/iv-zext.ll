; RUN: opt < %s -indvars -S | FileCheck %s
; CHECK-NOT: and
; CHECK-NOT: zext

target datalayout = "p:64:64:64-n32:64"

define void @foo(double* %d, i64 %n) nounwind {
entry:
	br label %loop

loop:
	%indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop ]
	%indvar.i8 = and i64 %indvar, 255
	%t0 = getelementptr double, double* %d, i64 %indvar.i8
	%t1 = load double, double* %t0
	%t2 = fmul double %t1, 0.1
	store double %t2, double* %t0
	%indvar.i24 = and i64 %indvar, 16777215
	%t3 = getelementptr double, double* %d, i64 %indvar.i24
	%t4 = load double, double* %t3
	%t5 = fmul double %t4, 2.3
	store double %t5, double* %t3
	%t6 = getelementptr double, double* %d, i64 %indvar
	%t7 = load double, double* %t6
	%t8 = fmul double %t7, 4.5
	store double %t8, double* %t6
	%indvar.next = add i64 %indvar, 1
	%exitcond = icmp eq i64 %indvar.next, 10
	br i1 %exitcond, label %return, label %loop

return:
	ret void
}
