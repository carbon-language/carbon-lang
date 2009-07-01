; RUN: llvm-as < %s | llc | grep {bl L_exact_log2.stub}
; PR4482
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "powerpc-apple-darwin8"

define i32 @foo(i64 %x) nounwind {
entry:
	%x_addr = alloca i64		; <i64*> [#uses=2]
	%retval = alloca i32		; <i32*> [#uses=2]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i64 %x, i64* %x_addr
	%1 = load i64* %x_addr, align 8		; <i64> [#uses=1]
	%2 = call i32 @exact_log2(i64 %1) nounwind		; <i32> [#uses=1]
	store i32 %2, i32* %0, align 4
	%3 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %3, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval1
}

define available_externally i32 @exact_log2(i64 %x) nounwind {
entry:
	%x_addr = alloca i64		; <i64*> [#uses=6]
	%retval = alloca i32		; <i32*> [#uses=2]
	%iftmp.0 = alloca i32		; <i32*> [#uses=3]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i64 %x, i64* %x_addr
	%1 = load i64* %x_addr, align 8		; <i64> [#uses=1]
	%2 = sub i64 0, %1		; <i64> [#uses=1]
	%3 = load i64* %x_addr, align 8		; <i64> [#uses=1]
	%4 = and i64 %2, %3		; <i64> [#uses=1]
	%5 = load i64* %x_addr, align 8		; <i64> [#uses=1]
	%6 = icmp ne i64 %4, %5		; <i1> [#uses=1]
	br i1 %6, label %bb2, label %bb

bb:		; preds = %entry
	%7 = load i64* %x_addr, align 8		; <i64> [#uses=1]
	%8 = icmp eq i64 %7, 0		; <i1> [#uses=1]
	br i1 %8, label %bb2, label %bb1

bb1:		; preds = %bb
	%9 = load i64* %x_addr, align 8		; <i64> [#uses=1]
	%10 = call i64 @llvm.cttz.i64(i64 %9)		; <i64> [#uses=1]
	%11 = trunc i64 %10 to i32		; <i32> [#uses=1]
	store i32 %11, i32* %iftmp.0, align 4
	br label %bb3

bb2:		; preds = %bb, %entry
	store i32 -1, i32* %iftmp.0, align 4
	br label %bb3

bb3:		; preds = %bb2, %bb1
	%12 = load i32* %iftmp.0, align 4		; <i32> [#uses=1]
	store i32 %12, i32* %0, align 4
	%13 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %13, i32* %retval, align 4
	br label %return

return:		; preds = %bb3
	%retval4 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval4
}

declare i64 @llvm.cttz.i64(i64) nounwind readnone
