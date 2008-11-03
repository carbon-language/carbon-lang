; RUN: llvm-as < %s | opt -loop-unswitch -stats | not grep loop-unswitch


define void @test_fc_while_continue_or(float %x, float %y, float* %result) nounwind {
entry:
	br label %bb2.outer

bb:		; preds = %bb2
	%0 = add float %5, %z.0		; <float> [#uses=3]
	%1 = fcmp oeq float %0, 0.000000e+00		; <i1> [#uses=1]
	br i1 %1, label %bb2, label %bb1

bb1:		; preds = %bb
	%.lcssa = phi float [ %0, %bb ]		; <float> [#uses=1]
	%z.0.lcssa1 = phi float [ %z.0, %bb ]		; <float> [#uses=0]
	%2 = add float %x_addr.0.ph, 1.000000e+00		; <float> [#uses=1]
	br label %bb2.outer

bb2.outer:		; preds = %bb1, %entry
	%z.0.ph = phi float [ 0.000000e+00, %entry ], [ %.lcssa, %bb1 ]		; <float> [#uses=1]
	%x_addr.0.ph = phi float [ %x, %entry ], [ %2, %bb1 ]		; <float> [#uses=3]
	%3 = fcmp une float %x_addr.0.ph, 0.000000e+00		; <i1> [#uses=1]
	%4 = fcmp une float %y, 0.000000e+00		; <i1> [#uses=1]
	%or.cond = or i1 %3, %4		; <i1> [#uses=1]
	%5 = mul float %x_addr.0.ph, %y		; <float> [#uses=1]
	br label %bb2

bb2:		; preds = %bb2.outer, %bb
	%z.0 = phi float [ %0, %bb ], [ %z.0.ph, %bb2.outer ]		; <float> [#uses=3]
	br i1 %or.cond, label %bb, label %bb4

bb4:		; preds = %bb2
	%z.0.lcssa = phi float [ %z.0, %bb2 ]		; <float> [#uses=1]
	store float %z.0.lcssa, float* %result, align 4
	ret void
}

define i32 @main() nounwind {
entry:
	%z = alloca [10 x i32]		; <[10 x i32]*> [#uses=2]
	%0 = call i32 (...)* @test_fc_while_or(i32 0, i32 0, [10 x i32]* %z) nounwind		; <i32> [#uses=0]
	%1 = getelementptr [10 x i32]* %z, i32 0, i32 0		; <i32*> [#uses=1]
	%2 = load i32* %1, align 4		; <i32> [#uses=1]
	ret i32 %2
}

declare i32 @test_fc_while_or(...)
