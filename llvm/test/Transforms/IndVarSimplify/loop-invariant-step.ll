; RUN: opt < %s -loop-index-split -instcombine -indvars -disable-output
; PR4455

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

declare i8* @fast_memcpy(i8*, i8*, i64)

define void @dvdsub_decode() nounwind {
entry:		; preds = %bb1
	br label %LoopA

LoopA:		; preds = %LoopA, %entry
	%x1.0.i17 = phi i32 [ %t0, %LoopA ], [ 0, %entry ]		; <i32> [#uses=2]
	%t0 = add i32 %x1.0.i17, 1		; <i32> [#uses=1]
	br i1 undef, label %LoopA, label %middle

middle:		; preds = %LoopA
	%t1 = sub i32 0, %x1.0.i17		; <i32> [#uses=1]
	%t2 = add i32 %t1, 1		; <i32> [#uses=1]
	br label %LoopB

LoopB:		; preds = %LoopB, %bb.nph.i27
	%y.029.i = phi i32 [ 0, %middle ], [ %t7, %LoopB ]		; <i32> [#uses=2]
	%t3 = mul i32 %y.029.i, %t2		; <i32> [#uses=1]
	%t4 = sext i32 %t3 to i64		; <i64> [#uses=1]
	%t5 = getelementptr i8* null, i64 %t4		; <i8*> [#uses=1]
	%t6 = call i8* @fast_memcpy(i8* %t5, i8* undef, i64 undef) nounwind		; <i8*> [#uses=0]
	%t7 = add i32 %y.029.i, 1		; <i32> [#uses=1]
	br i1 undef, label %LoopB, label %exit

exit:
	ret void
}
