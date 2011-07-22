; RUN: llc < %s -mattr=+avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @try_([2 x <8 x float>]* noalias %incarray, [2 x <8 x float>]* noalias %incarrayb ) {
enmtry:
	%incarray1 = alloca [2 x <8 x float>]*, align 8
	%incarrayb1 = alloca [2 x <8 x float>]*, align 8
	%carray = alloca [2 x <8 x float>], align 16
	%r = getelementptr [2 x <8 x float>]* %incarray, i32 0, i32 0
	%rb = getelementptr [2 x <8 x float>]* %incarrayb, i32 0, i32 0
	%r3 = load <8 x float>* %r, align 8
	%r4 = load <8 x float>* %rb, align 8
	%r8 = shufflevector <8 x float> %r3, <8 x float> %r4, <8 x i32> < i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11 >		; <<8 x float>> [#uses=1]
; CHECK-NOT: vunpcklps %ymm
	%r9 = getelementptr [2 x <8 x float>]* %carray, i32 0, i32 0
	store <8 x float> %r8, <8 x float>* %r9, align 4
	ret void
}
