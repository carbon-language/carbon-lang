; The i induction variable looks like a wrap-around, but it really is just
; a simple affine IV.  Make sure that indvars eliminates it.

; RUN: llvm-as < %s | opt -indvars | llvm-dis | grep phi | wc -l | grep 1

void %foo() {
entry:
	br label %bb6

bb6:		; preds = %cond_true, %entry
	%j.0 = phi int [ 1, %entry ], [ %tmp5, %cond_true ]		; <int> [#uses=3]
	%i.0 = phi int [ 0, %entry ], [ %j.0, %cond_true ]		; <int> [#uses=1]
	%tmp7 = call int (...)* %foo2( )		; <int> [#uses=1]
	%tmp = setne int %tmp7, 0		; <bool> [#uses=1]
	br bool %tmp, label %cond_true, label %return

cond_true:		; preds = %bb6
	%tmp2 = call int (...)* %bar( int %i.0, int %j.0 )		; <int> [#uses=0]
	%tmp5 = add int %j.0, 1		; <int> [#uses=1]
	br label %bb6

return:		; preds = %bb6
	ret void
}

declare int %bar(...)

declare int %foo2(...)
