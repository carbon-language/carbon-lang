; RUN: llvm-as < %s | opt -indvars
; PR4054

; Don't treat an and with 0 as a mask (trunc+zext).

define i32 @int80(i8 signext %p_71) nounwind {
entry:
	br label %bb

bb:		; preds = %bb6, %entry
	%p_71_addr.0 = phi i8 [ %p_71, %entry ], [ %0, %bb6 ]		; <i8> [#uses=0]
	br i1 false, label %bb4, label %bb1

bb1:		; preds = %bb
	ret i32 0

bb4:		; preds = %bb4, %bb
	br i1 false, label %bb6, label %bb4

bb6:		; preds = %bb4
	%0 = and i8 0, 0		; <i8> [#uses=1]
	br label %bb
}
