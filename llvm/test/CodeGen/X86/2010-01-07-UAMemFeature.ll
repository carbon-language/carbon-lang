; RUN: llc -mattr=vector-unaligned-mem < %s | FileCheck %s
; CHECK: addps{{[ \t]+}}(

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32 %n1, float* %A2, float* %B3, float* %C4) {
"file loop.c, line 1, bb1":	; srcLine 1
	%"$LCS_4" = alloca i64, align 8		; <i64*> [#uses=5]	; [oox.86 : sln.1]
	%"$LCS_5" = alloca i64, align 8		; <i64*> [#uses=5]	; [oox.87 : sln.1]
	%"$LCS_6" = alloca i64, align 8		; <i64*> [#uses=5]	; [oox.88 : sln.1]

	%r128 = load i64* %"$LCS_4", align 8		; <i64> [#uses=1]	; [oox.192 : sln.6]
	%r129 = inttoptr i64 %r128 to <4 x float>*		; <<4 x float>*> [#uses=1]	; [oox.192 : sln.6]
	%r130 = load <4 x float>* %r129, align 4		; <<4 x float>> [#uses=1]	; [oox.192 : sln.6]
	%r131 = load i64* %"$LCS_5", align 8		; <i64> [#uses=1]	; [oox.192 : sln.6]
	%r132 = inttoptr i64 %r131 to <4 x float>*		; <<4 x float>*> [#uses=1]	; [oox.192 : sln.6]
	%r133 = load <4 x float>* %r132, align 4		; <<4 x float>> [#uses=1]	; [oox.192 : sln.6]
	%r134 = add <4 x float> %r130, %r133		; <<4 x float>> [#uses=1]	; [oox.192 : sln.6]
	%r135 = load i64* %"$LCS_6", align 8		; <i64> [#uses=1]	; [oox.192 : sln.6]
	%r136 = inttoptr i64 %r135 to <4 x float>*		; <<4 x float>*> [#uses=1]	; [oox.192 : sln.6]
	store <4 x float> %r134, <4 x float>* %r136, align 4	; [oox.192 : sln.6]
	ret i32 0	; [oox.189 : sln.10]
}
