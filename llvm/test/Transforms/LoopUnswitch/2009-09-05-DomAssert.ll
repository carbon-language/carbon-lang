; RUN: llvm-as < %s | opt -loop-unswitch -disable-output 
; rdar://7197574

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-apple-darwin9"
	%struct.frame = type { i16*, i16*, i16* }

declare arm_apcscc i32 @ercCollect8PredBlocks(i32* nocapture, i32, i32, i32* nocapture, i32, i32, i32, i8 zeroext) nounwind

define arm_apcscc void @concealBlocks(i32 %lastColumn, i32 %lastRow, i32 %comp, %struct.frame* nocapture %recfr, i32 %picSizeX, i32* nocapture %condition) nounwind {
entry:
	br i1 undef, label %bb.nph12, label %return

bb28:		; preds = %bb.nph12
	unreachable

bb42:		; preds = %bb.nph12
	br label %bb43

bb43:		; preds = %bb61, %bb42
	%0 = call arm_apcscc  i32 @ercCollect8PredBlocks(i32* undef, i32 undef, i32 0, i32* %condition, i32 %lastRow, i32 %lastColumn, i32 undef, i8 zeroext 1) nounwind		; <i32> [#uses=0]
	switch i32 %comp, label %bb58 [
		i32 0, label %bb52
		i32 1, label %bb54
		i32 2, label %bb56
	]

bb52:		; preds = %bb43
	br label %bb58

bb54:		; preds = %bb43
	br label %bb58

bb56:		; preds = %bb43
	unreachable

bb58:		; preds = %bb54, %bb52, %bb43
	br i1 %1, label %bb59, label %bb61

bb59:		; preds = %bb58
	br label %bb61

bb61:		; preds = %bb59, %bb58
	br label %bb43

bb.nph12:		; preds = %entry
	%1 = icmp eq i32 %comp, 0		; <i1> [#uses=1]
	br i1 undef, label %bb28, label %bb42

return:		; preds = %entry
	ret void
}
