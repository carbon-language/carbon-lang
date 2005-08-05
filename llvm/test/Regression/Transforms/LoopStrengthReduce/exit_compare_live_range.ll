; Make sure that the compare instruction occurs after the increment to avoid
; having overlapping live ranges that result in copies.  We want the setcc instruction
; immediately before the conditional branch.
;
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | %prcontext 'br bool' 1 | grep set
; XFAIL: *

void %foo(float* %D, uint %E) {
entry:
	br label %no_exit

no_exit:
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %no_exit ]
	volatile store float 0.0, float* %D
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=2]
	%exitcond = seteq uint %indvar.next, %E		; <bool> [#uses=1]
	br bool %exitcond, label %loopexit, label %no_exit

loopexit:
	ret void
}
