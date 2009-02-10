; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin | not grep fstpt
; PR3457
; rdar://6548010

define void @foo(double* nocapture %P) nounwind {
entry:
	%0 = tail call double (...)* @test() nounwind		; <double> [#uses=2]
	%1 = tail call double (...)* @test() nounwind		; <double> [#uses=2]
	%2 = mul double %0, %0		; <double> [#uses=1]
	%3 = mul double %1, %1		; <double> [#uses=1]
	%4 = add double %2, %3		; <double> [#uses=1]
	store double %4, double* %P, align 8
	ret void
}

declare double @test(...)
