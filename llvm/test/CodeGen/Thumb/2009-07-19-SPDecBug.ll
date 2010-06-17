; RUN: llc < %s -mtriple=thumbv6-elf | not grep "subs sp"
; PR4567

define i8* @__gets_chk(i8* %s, i32 %slen) nounwind {
entry:
	br i1 undef, label %bb, label %bb1

bb:		; preds = %entry
	ret i8* undef

bb1:		; preds = %entry
	br i1 undef, label %bb3, label %bb2

bb2:		; preds = %bb1
	%0 = alloca i8, i32 undef, align 4		; <i8*> [#uses=0]
	br label %bb4

bb3:		; preds = %bb1
	%1 = malloc i8, i32 undef		; <i8*> [#uses=0]
	br label %bb4

bb4:		; preds = %bb3, %bb2
	br i1 undef, label %bb5, label %bb6

bb5:		; preds = %bb4
	%2 = call  i8* @gets(i8* %s) nounwind		; <i8*> [#uses=1]
	ret i8* %2

bb6:		; preds = %bb4
	unreachable
}

declare i8* @gets(i8*) nounwind
