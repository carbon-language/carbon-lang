; Testcase reduced from 197.parser by bugpoint
; RUN: llvm-as < %s | opt -adce 
implementation   ; Functions:

void %conjunction_prune() {
; <label>:0		; No predecessors!
	br label %bb19

bb19:		; preds = %bb22, %bb23, %0
	%reg205 = phi sbyte* [ null, %bb22 ], [ null, %bb23 ], [ null, %0 ]		; <sbyte*> [#uses=1]
	br bool false, label %bb21, label %bb22

bb21:		; preds = %bb19
	%cast455 = cast sbyte* %reg205 to sbyte**		; <sbyte**> [#uses=0]
	br label %bb22

bb22:		; preds = %bb21, %bb19
	br bool false, label %bb19, label %bb23

bb23:		; preds = %bb22
	br bool false, label %bb19, label %bb28

bb28:		; preds = %bb23
	ret void
}
