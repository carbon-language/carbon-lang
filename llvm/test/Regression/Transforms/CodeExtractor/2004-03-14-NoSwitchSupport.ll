; RUN: llvm-as < %s | opt -loop-extract-single -disable-output

void %ab() {
entry:
	br label %codeReplTail

then.1:		; preds = %codeReplTail
	br label %loopentry.1

loopentry.1:		; preds = %loopentry.1.preheader, %no_exit.1
	br bool false, label %no_exit.1, label %loopexit.0.loopexit1

no_exit.1:		; preds = %loopentry.1
	br label %loopentry.1

loopexit.0.loopexit:		; preds = %codeReplTail
	ret void

loopexit.0.loopexit1:		; preds = %loopentry.1
	ret void

codeReplTail:		; preds = %codeRepl, %codeReplTail
	switch ushort 0, label %codeReplTail [
		 ushort 0, label %loopexit.0.loopexit
		 ushort 1, label %then.1
	]
}
