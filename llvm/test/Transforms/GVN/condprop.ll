; RUN: opt < %s -basicaa -gvn -S | grep {br i1 false}

@a = external global i32		; <i32*> [#uses=7]

define i32 @foo() nounwind {
entry:
	%0 = load i32* @a, align 4		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 4		; <i1> [#uses=1]
	br i1 %1, label %bb, label %bb1

bb:		; preds = %entry
	br label %bb8

bb1:		; preds = %entry
	%2 = load i32* @a, align 4		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 5		; <i1> [#uses=1]
	br i1 %3, label %bb2, label %bb3

bb2:		; preds = %bb1
	br label %bb8

bb3:		; preds = %bb1
	%4 = load i32* @a, align 4		; <i32> [#uses=1]
	%5 = icmp eq i32 %4, 4		; <i1> [#uses=1]
	br i1 %5, label %bb4, label %bb5

bb4:		; preds = %bb3
	%6 = load i32* @a, align 4		; <i32> [#uses=1]
	%7 = add i32 %6, 5		; <i32> [#uses=1]
	br label %bb8

bb5:		; preds = %bb3
	%8 = load i32* @a, align 4		; <i32> [#uses=1]
	%9 = icmp eq i32 %8, 5		; <i1> [#uses=1]
	br i1 %9, label %bb6, label %bb7

bb6:		; preds = %bb5
	%10 = load i32* @a, align 4		; <i32> [#uses=1]
	%11 = add i32 %10, 4		; <i32> [#uses=1]
	br label %bb8

bb7:		; preds = %bb5
	%12 = load i32* @a, align 4		; <i32> [#uses=1]
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb4, %bb2, %bb
	%.0 = phi i32 [ %12, %bb7 ], [ %11, %bb6 ], [ %7, %bb4 ], [ 4, %bb2 ], [ 5, %bb ]		; <i32> [#uses=1]
	br label %return

return:		; preds = %bb8
	ret i32 %.0
}
