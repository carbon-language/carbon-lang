; RUN: llc < %s -mtriple=i386-apple-darwin9 -stats |& grep regcoalescing | grep commuting

@lookupTable5B = external global [64 x i32], align 32		; <[64 x i32]*> [#uses=1]
@lookupTable3B = external global [16 x i32], align 32		; <[16 x i32]*> [#uses=1]
@disparity0 = external global i32		; <i32*> [#uses=5]
@disparity1 = external global i32		; <i32*> [#uses=3]

define i32 @calc(i32 %theWord, i32 %k) nounwind {
entry:
	%0 = lshr i32 %theWord, 3		; <i32> [#uses=1]
	%1 = and i32 %0, 31		; <i32> [#uses=1]
	%2 = shl i32 %k, 5		; <i32> [#uses=1]
	%3 = or i32 %1, %2		; <i32> [#uses=1]
	%4 = and i32 %theWord, 7		; <i32> [#uses=1]
	%5 = shl i32 %k, 3		; <i32> [#uses=1]
	%6 = or i32 %5, %4		; <i32> [#uses=1]
	%7 = getelementptr [64 x i32]* @lookupTable5B, i32 0, i32 %3		; <i32*> [#uses=1]
	%8 = load i32* %7, align 4		; <i32> [#uses=5]
	%9 = getelementptr [16 x i32]* @lookupTable3B, i32 0, i32 %6		; <i32*> [#uses=1]
	%10 = load i32* %9, align 4		; <i32> [#uses=5]
	%11 = and i32 %8, 65536		; <i32> [#uses=1]
	%12 = icmp eq i32 %11, 0		; <i1> [#uses=1]
	br i1 %12, label %bb1, label %bb

bb:		; preds = %entry
	%13 = and i32 %8, 994		; <i32> [#uses=1]
	%14 = load i32* @disparity0, align 4		; <i32> [#uses=2]
	store i32 %14, i32* @disparity1, align 4
	br label %bb8

bb1:		; preds = %entry
	%15 = lshr i32 %8, 18		; <i32> [#uses=1]
	%16 = and i32 %15, 1		; <i32> [#uses=1]
	%17 = load i32* @disparity0, align 4		; <i32> [#uses=4]
	%18 = icmp eq i32 %16, %17		; <i1> [#uses=1]
	%not = select i1 %18, i32 0, i32 994		; <i32> [#uses=1]
	%.masked = and i32 %8, 994		; <i32> [#uses=1]
	%result.1 = xor i32 %not, %.masked		; <i32> [#uses=2]
	%19 = and i32 %8, 524288		; <i32> [#uses=1]
	%20 = icmp eq i32 %19, 0		; <i1> [#uses=1]
	br i1 %20, label %bb7, label %bb6

bb6:		; preds = %bb1
	%21 = xor i32 %17, 1		; <i32> [#uses=2]
	store i32 %21, i32* @disparity1, align 4
	br label %bb8

bb7:		; preds = %bb1
	store i32 %17, i32* @disparity1, align 4
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb
	%22 = phi i32 [ %17, %bb7 ], [ %21, %bb6 ], [ %14, %bb ]		; <i32> [#uses=4]
	%result.0 = phi i32 [ %result.1, %bb7 ], [ %result.1, %bb6 ], [ %13, %bb ]		; <i32> [#uses=2]
	%23 = and i32 %10, 65536		; <i32> [#uses=1]
	%24 = icmp eq i32 %23, 0		; <i1> [#uses=1]
	br i1 %24, label %bb10, label %bb9

bb9:		; preds = %bb8
	%25 = and i32 %10, 29		; <i32> [#uses=1]
	%26 = or i32 %result.0, %25		; <i32> [#uses=1]
	store i32 %22, i32* @disparity0, align 4
	ret i32 %26

bb10:		; preds = %bb8
	%27 = lshr i32 %10, 18		; <i32> [#uses=1]
	%28 = and i32 %27, 1		; <i32> [#uses=1]
	%29 = icmp eq i32 %28, %22		; <i1> [#uses=1]
	%not13 = select i1 %29, i32 0, i32 29		; <i32> [#uses=1]
	%.masked20 = and i32 %10, 29		; <i32> [#uses=1]
	%.pn = xor i32 %not13, %.masked20		; <i32> [#uses=1]
	%result.3 = or i32 %.pn, %result.0		; <i32> [#uses=2]
	%30 = and i32 %10, 524288		; <i32> [#uses=1]
	%31 = icmp eq i32 %30, 0		; <i1> [#uses=1]
	br i1 %31, label %bb17, label %bb16

bb16:		; preds = %bb10
	%32 = xor i32 %22, 1		; <i32> [#uses=1]
	store i32 %32, i32* @disparity0, align 4
	ret i32 %result.3

bb17:		; preds = %bb10
	store i32 %22, i32* @disparity0, align 4
	ret i32 %result.3
}
