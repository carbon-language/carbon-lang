; RUN: llvm-as < %s | opt -lowerswitch -disable-output

void %solve() {
entry:
	%targetBlock = call ushort %solve_code( )		; <ushort> [#uses=1]
	br label %codeReplTail

then.1:		; preds = %codeReplTail
	ret void

loopexit.0:		; preds = %codeReplTail
	ret void

codeReplTail:		; preds = %entry, %codeReplTail
	switch ushort %targetBlock, label %codeReplTail [
		 ushort 0, label %loopexit.0
		 ushort 1, label %then.1
	]
}

declare ushort %solve_code()
