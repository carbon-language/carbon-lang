; RUN: opt < %s -ssi-everything -disable-output

declare fastcc i32 @ras_Empty(i8** nocapture) nounwind readonly

define i32 @cc_Tautology() nounwind {
entry:
	unreachable

cc_InitData.exit:		; No predecessors!
	%0 = call fastcc i32 @ras_Empty(i8** undef) nounwind		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb2, label %bb6

bb2:		; preds = %cc_InitData.exit
	unreachable

bb6:		; preds = %cc_InitData.exit
	ret i32 undef
}
