; RUN: llvm-as < %s | opt -loop-extract -disable-output


void %maketree() {
entry:
	br bool false, label %no_exit.1, label %loopexit.0

no_exit.1:		; preds = %entry, %expandbox.entry, %endif
	br bool false, label %endif, label %expandbox.entry

expandbox.entry:		; preds = %no_exit.1
	br bool false, label %loopexit.1, label %no_exit.1

endif:		; preds = %no_exit.1
	br bool false, label %loopexit.1, label %no_exit.1

loopexit.1:		; preds = %expandbox.entry, %endif
	%ic.i.0.0.4 = phi int [ 0, %expandbox.entry ], [ 0, %endif ]		; <int> [#uses=0]
	ret void

loopexit.0:		; preds = %entry
	ret void
}
