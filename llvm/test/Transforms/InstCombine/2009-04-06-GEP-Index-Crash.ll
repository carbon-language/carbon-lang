; RUN: llvm-as < %s | opt -instcombine | llvm-dis
; rdar://6762290

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
	%T = type <{ i64, i64, i64 }>

define i32 @test(i8* %start, i32 %X) nounwind {
entry:
	%tmp3 = load i64* null		; <i64> [#uses=1]
	%add.ptr = getelementptr i8* %start, i64 %tmp3		; <i8*> [#uses=1]
	%tmp158 = load i32* null		; <i32> [#uses=1]
	%add.ptr159 = getelementptr %T* null, i32 %tmp158
	%add.ptr209 = getelementptr i8* %start, i64 0		; <i8*> [#uses=1]
	%add.ptr212 = getelementptr i8* %add.ptr209, i32 %X		; <i8*> [#uses=1]
	%cmp214 = icmp ugt i8* %add.ptr212, %add.ptr		; <i1> [#uses=1]
	br i1 %cmp214, label %if.then216, label %if.end363

if.then216:		; preds = %for.body162
	ret i32 1

if.end363:		; preds = %for.body162
	ret i32 0
}
