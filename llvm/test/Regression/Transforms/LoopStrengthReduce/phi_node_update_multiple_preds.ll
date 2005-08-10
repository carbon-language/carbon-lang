; RUN: llvm-as < %s | opt -loop-reduce -disable-output
; LSR should not crash on this.

fastcc void %loadloop() {
entry:
	switch sbyte 0, label %shortcirc_next [
		 sbyte 32, label %loopexit.2
		 sbyte 59, label %loopexit.2
	]

shortcirc_next:		; preds = %no_exit.2, %entry
	%indvar37 = phi uint [ 0, %entry ], [ %indvar.next38, %no_exit.2 ]		; <uint> [#uses=3]
	%wp.2.4 = getelementptr sbyte* null, uint %indvar37		; <sbyte*> [#uses=1]
	br bool false, label %loopexit.2, label %no_exit.2

no_exit.2:		; preds = %shortcirc_next
	%wp.2.4.rec = cast uint %indvar37 to int		; <int> [#uses=1]
	%inc.1.rec = add int %wp.2.4.rec, 1		; <int> [#uses=1]
	%inc.1 = getelementptr sbyte* null, int %inc.1.rec		; <sbyte*> [#uses=2]
	%indvar.next38 = add uint %indvar37, 1		; <uint> [#uses=1]
	switch sbyte 0, label %shortcirc_next [
		 sbyte 32, label %loopexit.2
		 sbyte 59, label %loopexit.2
	]

loopexit.2:		; preds = %no_exit.2, %no_exit.2, %shortcirc_next, %entry, %entry
	%wp.2.7 = phi sbyte* [ null, %entry ], [ null, %entry ], [ %wp.2.4, %shortcirc_next ], [ %inc.1, %no_exit.2 ], [ %inc.1, %no_exit.2 ]		; <sbyte*> [#uses=0]
	ret void
}
