; RUN: llvm-as < %s | llc | not grep movss
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

; This should store directly into P from the FP stack.  It should not
; go through a stack slot to get there.

define void @bar(double* %P) {
entry:
	%tmp = tail call double (...)* @foo( )		; <double> [#uses=1]
	store double %tmp, double* %P, align 8
	ret void
}

declare double @foo(...)
