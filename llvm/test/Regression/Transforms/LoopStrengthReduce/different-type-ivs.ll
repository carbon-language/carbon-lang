; RUN: llvm-as < %s | opt -loop-reduce -disable-output
; Test to make sure that loop-reduce never crashes on IV's 
; with different types but identical strides.

void %foo() {
entry:
	br label %no_exit

no_exit:		; preds = %no_exit, %entry
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %no_exit ]		; <uint> [#uses=3]
	%indvar = cast uint %indvar to short		; <short> [#uses=1]
	%X.0.0 = mul short %indvar, 1234		; <short> [#uses=1]
	%tmp. = mul uint %indvar, 1234		; <uint> [#uses=1]
	%tmp.5 = cast short %X.0.0 to int		; <int> [#uses=1]
	%tmp.3 = call int (...)* %bar( int %tmp.5, uint %tmp. )		; <int> [#uses=0]
	%tmp.0 = call bool %pred( )		; <int> [#uses=1]
	%indvar.next = add uint %indvar, 1		; <uint> [#uses=1]
	br bool %tmp.0, label %return, label %no_exit

return:
	ret void
}

declare bool %pred()

declare int %bar(...)
