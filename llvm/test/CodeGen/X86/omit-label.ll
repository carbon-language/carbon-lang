; RUN: llvm-as < %s | llc -march=x86-64 | grep BB1_1:
; PR4126

; Don't omit this label's definition.

define void @bux(i32 %p_53) nounwind optsize {
entry:
	%0 = icmp eq i32 %p_53, 0		; <i1> [#uses=1]
	%1 = icmp sgt i32 %p_53, 0		; <i1> [#uses=1]
	%or.cond = and i1 %0, %1		; <i1> [#uses=1]
	br i1 %or.cond, label %bb.i, label %bb3

bb.i:		; preds = %entry
	%2 = add i32 %p_53, 1		; <i32> [#uses=1]
	%3 = icmp slt i32 %2, 0		; <i1> [#uses=0]
	br label %bb3

bb3:		; preds = %bb.i, %entry
	%4 = tail call i32 (...)* @baz(i32 0) nounwind		; <i32> [#uses=0]
	ret void
}

declare i32 @baz(...)
