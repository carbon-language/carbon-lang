; This testcase tests to see if adds and PHI's are handled in their full 
; generality.  This testcase comes from the following C code:
;
; void f() {
;   int i;
;   for (i = 1; i < 100; i++) {
;     if (i)
;       g();
;   }
; }
;
; Note that this is a "feature" test, not a correctness test.
;
; RUN: llvm-as < %s | opt -cee -simplifycfg | llvm-dis | not grep cond213
;
implementation   ; Functions:

declare void %g()

void %f() {
bb0:		; No predecessors!
	br label %bb2

bb2:		; preds = %bb4, %bb0
	%cann-indvar = phi int [ 0, %bb0 ], [ %add1-indvar, %bb4 ]		; <int> [#uses=2]
	%add1-indvar = add int %cann-indvar, 1		; <int> [#uses=2]
	%cond213 = seteq int %add1-indvar, 0		; <bool> [#uses=1]
	br bool %cond213, label %bb4, label %bb3

bb3:		; preds = %bb2
	call void %g( )
	br label %bb4

bb4:		; preds = %bb3, %bb2
	%reg109 = add int %cann-indvar, 2		; <int> [#uses=1]
	%cond217 = setle int %reg109, 99		; <bool> [#uses=1]
	br bool %cond217, label %bb2, label %bb5

bb5:		; preds = %bb4
	ret void
}
