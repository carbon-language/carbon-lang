; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep phi | wc -l | grep 1

; This should only result in one PHI node!
; XFAIL: *

; void foo(double *D, double *E, double F) {
;   while (D != E)
;     *D++ = F;
; }

void %foo(double* %D, double* %E, double %F) {
entry:
	%tmp.24 = seteq double* %D, %E		; <bool> [#uses=1]
	br bool %tmp.24, label %return, label %no_exit

no_exit:		; preds = %no_exit, %entry
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %no_exit ]		; <uint> [#uses=3]
	%D_addr.0.0.rec = cast uint %indvar to int		; <int> [#uses=1]
	%D_addr.0.0 = getelementptr double* %D, uint %indvar		; <double*> [#uses=1]
	%inc.rec = add int %D_addr.0.0.rec, 1		; <int> [#uses=1]
	%inc = getelementptr double* %D, int %inc.rec		; <double*> [#uses=1]
	store double %F, double* %D_addr.0.0
	%tmp.2 = seteq double* %inc, %E		; <bool> [#uses=1]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	br bool %tmp.2, label %return, label %no_exit

return:		; preds = %no_exit, %entry
	ret void
}
