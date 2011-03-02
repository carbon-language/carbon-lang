; RUN: llc < %s -mattr=+avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @try_([2 x <4 x double>]* noalias %incarray, [2 x <4 x double>]* noalias %incarrayb ) {
entry:
	%incarray1 = alloca [2 x <4 x double>]*, align 8
	%incarrayb1 = alloca [2 x <4 x double>]*, align 8
	%carray = alloca [2 x <4 x double>], align 16
	%r = getelementptr [2 x <4 x double>]* %incarray, i32 0, i32 0
	%rb = getelementptr [2 x <4 x double>]* %incarrayb, i32 0, i32 0
	%r3 = load <4 x double>* %r, align 8
	%r4 = load <4 x double>* %rb, align 8
	%r11 = shufflevector <4 x double> %r3, <4 x double> %r4, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x double>> [#uses=1]
; CHECK-NOT: vunpcklpd
	%r12 = getelementptr [2 x <4 x double>]* %carray, i32 0, i32 1
	store <4 x double> %r11, <4 x double>* %r12, align 4
	ret void
}
