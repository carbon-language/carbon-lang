; RUN: llvm-as < %s | opt -loop-reduce -disable-output

void %main() {
entry:
	br label %loopentry.0

loopentry.0:		; preds = %then.5, %entry
	%arg_index.1.ph = phi int [ 1, %entry ], [ %arg_index.1.ph.be, %then.5 ]		; <int> [#uses=1]
	br bool false, label %no_exit.0, label %loopexit.0

no_exit.0:		; preds = %loopentry.0
	%arg_index.1.1 = add int 0, %arg_index.1.ph		; <int> [#uses=2]
	br bool false, label %then.i55, label %endif.i61

then.i55:		; preds = %no_exit.0
	br bool false, label %then.4, label %else.1

endif.i61:		; preds = %no_exit.0
	ret void

then.4:		; preds = %then.i55
	%tmp.19993 = add int %arg_index.1.1, 2		; <int> [#uses=0]
	ret void

else.1:		; preds = %then.i55
	br bool false, label %then.i86, label %loopexit.i97

then.i86:		; preds = %else.1
	ret void

loopexit.i97:		; preds = %else.1
	br bool false, label %then.5, label %else.2

then.5:		; preds = %loopexit.i97
	%arg_index.1.ph.be = add int %arg_index.1.1, 2		; <int> [#uses=1]
	br label %loopentry.0

else.2:		; preds = %loopexit.i97
	ret void

loopexit.0:		; preds = %loopentry.0
	ret void
}
