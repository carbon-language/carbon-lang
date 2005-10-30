; RUN: llvm-as < %s | opt -loop-reduce &&
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | not grep mul

; Make sure we don't get a multiply by 6 in this loop.

int %foo(int %A, int %B, int %C, int %D) {
entry:
	%tmp.5 = setgt int %C, 0		; <bool> [#uses=1]
	%tmp.25 = and int %A, 1		; <int> [#uses=1]
	br label %loopentry.1

loopentry.1:		; preds = %loopexit.1, %entry
	%indvar20 = phi uint [ 0, %entry ], [ %indvar.next21, %loopexit.1 ]		; <uint> [#uses=2]
	%k.1 = phi int [ 0, %entry ], [ %k.1.3, %loopexit.1 ]		; <int> [#uses=2]
	br bool %tmp.5, label %no_exit.1.preheader, label %loopexit.1

no_exit.1.preheader:		; preds = %loopentry.1
	%i.0.0 = cast uint %indvar20 to int		; <int> [#uses=1]
	%tmp.9 = mul int %i.0.0, 6		; <int> [#uses=1]
	br label %no_exit.1.outer

no_exit.1.outer:		; preds = %cond_true, %no_exit.1.preheader
	%k.1.2.ph = phi int [ %k.1, %no_exit.1.preheader ], [ %k.09, %cond_true ]		; <int> [#uses=2]
	%j.1.2.ph = phi int [ 0, %no_exit.1.preheader ], [ %inc.1, %cond_true ]		; <int> [#uses=1]
	br label %no_exit.1

no_exit.1:		; preds = %cond_continue, %no_exit.1.outer
	%indvar = phi uint [ 0, %no_exit.1.outer ], [ %indvar.next, %cond_continue ]		; <uint> [#uses=2]
	%indvar = cast uint %indvar to int		; <int> [#uses=1]
	%j.1.2 = add int %indvar, %j.1.2.ph		; <int> [#uses=2]
	%tmp.11 = add int %j.1.2, %tmp.9		; <int> [#uses=1]
	%tmp.12 = cast int %tmp.11 to ubyte		; <ubyte> [#uses=1]
	%tmp.13 = shl int %D, ubyte %tmp.12		; <int> [#uses=2]
	%tmp.15 = seteq int %tmp.13, %B		; <bool> [#uses=1]
	%inc.1 = add int %j.1.2, 1		; <int> [#uses=3]
	br bool %tmp.15, label %cond_true, label %cond_continue

cond_true:		; preds = %no_exit.1
	%tmp.26 = and int %tmp.25, %tmp.13		; <int> [#uses=1]
	%k.09 = add int %tmp.26, %k.1.2.ph		; <int> [#uses=2]
	%tmp.517 = setlt int %inc.1, %C		; <bool> [#uses=1]
	br bool %tmp.517, label %no_exit.1.outer, label %loopexit.1

cond_continue:		; preds = %no_exit.1
	%tmp.519 = setlt int %inc.1, %C		; <bool> [#uses=1]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	br bool %tmp.519, label %no_exit.1, label %loopexit.1

loopexit.1:		; preds = %cond_continue, %cond_true, %loopentry.1
	%k.1.3 = phi int [ %k.1, %loopentry.1 ], [ %k.09, %cond_true ], [ %k.1.2.ph, %cond_continue ]		; <int> [#uses=2]
	%indvar.next21 = add uint %indvar20, 1		; <uint> [#uses=2]
	%exitcond = seteq uint %indvar.next21, 4		; <bool> [#uses=1]
	br bool %exitcond, label %loopexit.0, label %loopentry.1

loopexit.0:		; preds = %loopexit.1
	ret int %k.1.3
}
