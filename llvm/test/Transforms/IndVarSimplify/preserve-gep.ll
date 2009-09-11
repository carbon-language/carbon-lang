; RUN: opt < %s -indvars -S > %t
; RUN: not grep ptrtoint %t
; RUN: not grep inttoptr %t
; RUN: grep getelementptr %t | count 1

; Indvars shouldn't leave getelementptrs expanded out as
; inttoptr+ptrtoint in its output in common cases.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.Foo = type { i32, i32, [10 x i32], i32 }

define void @me(%struct.Foo* nocapture %Bar) nounwind {
entry:
	br i1 false, label %return, label %bb.nph

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%i.01 = phi i64 [ %4, %bb1 ], [ 0, %bb.nph ]		; <i64> [#uses=3]
	%0 = getelementptr %struct.Foo* %Bar, i64 %i.01, i32 2, i64 3		; <i32*> [#uses=1]
	%1 = load i32* %0, align 4		; <i32> [#uses=1]
	%2 = mul i32 %1, 113		; <i32> [#uses=1]
	%3 = getelementptr %struct.Foo* %Bar, i64 %i.01, i32 2, i64 3		; <i32*> [#uses=1]
	store i32 %2, i32* %3, align 4
	%4 = add i64 %i.01, 1		; <i64> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%phitmp = icmp sgt i64 %4, 19999		; <i1> [#uses=1]
	br i1 %phitmp, label %bb1.return_crit_edge, label %bb

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}
