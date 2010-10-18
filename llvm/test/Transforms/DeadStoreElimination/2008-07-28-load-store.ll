; RUN: opt < %s -basicaa -dse -S | not grep tmp5
; PR2599
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define void @foo({ i32, i32 }* %x) nounwind  {
entry:
	%tmp4 = getelementptr { i32, i32 }* %x, i32 0, i32 0		; <i32*> [#uses=2]
	%tmp5 = load i32* %tmp4, align 4		; <i32> [#uses=1]
	%tmp7 = getelementptr { i32, i32 }* %x, i32 0, i32 1		; <i32*> [#uses=2]
	%tmp8 = load i32* %tmp7, align 4		; <i32> [#uses=1]
	%tmp17 = sub i32 0, %tmp8		; <i32> [#uses=1]
	store i32 %tmp5, i32* %tmp4, align 4
	store i32 %tmp17, i32* %tmp7, align 4
	ret void
}
