; RUN: opt < %s -instcombine -S | not grep getelementptr
; PR2831

; Don't raise arbitrary inttoptr+arithmetic+ptrtoint to getelementptr.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	%0 = ptrtoint i8** %argv to i32		; <i32> [#uses=1]
	%1 = add i32 %0, 1		; <i32> [#uses=1]
	ret i32 %1
}

; This testcase could theoretically be optimized down to return zero,
; but for now being conservative with ptrtoint/inttoptr is fine.
define i32 @a() nounwind {
entry:
	%b = alloca i32		; <i32*> [#uses=3]
	%a = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 1, i32* %b, align 4
	%a1 = ptrtoint i32* %a to i32		; <i32> [#uses=1]
	%b4 = ptrtoint i32* %b to i32		; <i32> [#uses=1]
	%a7 = ptrtoint i32* %a to i32		; <i32> [#uses=1]
	%0 = sub i32 %b4, %a7		; <i32> [#uses=1]
	%1 = add i32 %a1, %0		; <i32> [#uses=1]
	%2 = inttoptr i32 %1 to i32*		; <i32*> [#uses=1]
	store i32 0, i32* %2, align 4
	%3 = load i32* %b, align 4		; <i32> [#uses=1]
	br label %return

return:		; preds = %entry
	ret i32 %3
}
