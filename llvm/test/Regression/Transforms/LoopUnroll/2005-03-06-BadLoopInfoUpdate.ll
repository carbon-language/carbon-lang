; RUN: llvm-as < %s | opt -loop-unroll -loopsimplify -disable-output

implementation   ; Functions:

void %print_board() {
entry:
	br label %no_exit.1

no_exit.1:		; preds = %cond_false.2, %entry
	br label %no_exit.2

no_exit.2:		; preds = %no_exit.2, %no_exit.1
	%indvar1 = phi uint [ 0, %no_exit.1 ], [ %indvar.next2, %no_exit.2 ]		; <uint> [#uses=1]
	%indvar.next2 = add uint %indvar1, 1		; <uint> [#uses=2]
	%exitcond3 = setne uint %indvar.next2, 7		; <bool> [#uses=1]
	br bool %exitcond3, label %no_exit.2, label %loopexit.2

loopexit.2:		; preds = %no_exit.2
	br bool false, label %cond_true.2, label %cond_false.2

cond_true.2:		; preds = %loopexit.2
	ret void

cond_false.2:		; preds = %loopexit.2
	br bool false, label %no_exit.1, label %loopexit.1

loopexit.1:		; preds = %cond_false.2
	ret void
}
