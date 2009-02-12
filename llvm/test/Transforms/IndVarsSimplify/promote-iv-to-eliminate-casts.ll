; RUN: llvm-as < %s | opt -indvars | llvm-dis | not grep sext

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define i64 @test(i64* nocapture %first, i32 %count) nounwind readonly {
entry:
	%t0 = icmp sgt i32 %count, 0		; <i1> [#uses=1]
	br i1 %t0, label %bb.nph, label %bb2

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%result.02 = phi i64 [ %t5, %bb1 ], [ 0, %bb.nph ]		; <i64> [#uses=1]
	%n.01 = phi i32 [ %t6, %bb1 ], [ 0, %bb.nph ]		; <i32> [#uses=2]
	%t1 = sext i32 %n.01 to i64		; <i64> [#uses=1]
	%t2 = getelementptr i64* %first, i64 %t1		; <i64*> [#uses=1]
	%t3 = load i64* %t2, align 8		; <i64> [#uses=1]
	%t4 = lshr i64 %t3, 4		; <i64> [#uses=1]
	%t5 = add i64 %t4, %result.02		; <i64> [#uses=2]
	%t6 = add i32 %n.01, 1		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%t7 = icmp slt i32 %t6, %count		; <i1> [#uses=1]
	br i1 %t7, label %bb, label %bb1.bb2_crit_edge

bb1.bb2_crit_edge:		; preds = %bb1
	%.lcssa = phi i64 [ %t5, %bb1 ]		; <i64> [#uses=1]
	br label %bb2

bb2:		; preds = %bb1.bb2_crit_edge, %entry
	%result.0.lcssa = phi i64 [ %.lcssa, %bb1.bb2_crit_edge ], [ 0, %entry ]		; <i64> [#uses=1]
	ret i64 %result.0.lcssa
}

define void @foo(i16 signext %N, i32* nocapture %P) nounwind {
entry:
	%t0 = icmp sgt i16 %N, 0		; <i1> [#uses=1]
	br i1 %t0, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%i.01 = phi i16 [ %t3, %bb1 ], [ 0, %bb.nph ]		; <i16> [#uses=2]
	%t1 = sext i16 %i.01 to i64		; <i64> [#uses=1]
	%t2 = getelementptr i32* %P, i64 %t1		; <i32*> [#uses=1]
	store i32 123, i32* %t2, align 4
	%t3 = add i16 %i.01, 1		; <i16> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%t4 = icmp slt i16 %t3, %N		; <i1> [#uses=1]
	br i1 %t4, label %bb, label %bb1.return_crit_edge

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}
