; RUN: opt < %s -gvn | llvm-dis

@bsLive = external global i32		; <i32*> [#uses=2]

define i32 @bsR(i32 %n) {
entry:
	br i1 false, label %cond_next, label %bb19

cond_next:		; preds = %entry
	store i32 0, i32* @bsLive, align 4
	br label %bb19

bb19:		; preds = %cond_next, %entry
	%tmp29 = load i32, i32* @bsLive, align 4		; <i32> [#uses=0]
	ret i32 0
}
