; RUN: llc < %s -march=x86 -mattr=sse41 -o %t
; RUN: grep addps %t | count 2
; RUN: grep mulps %t | count 2
; RUN: grep subps %t | count 2

; ModuleID = 'vec_shuffle-27.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-apple-cl.1.0"

define <8 x float> @my2filter4_1d(<4 x float> %a, <8 x float> %T0, <8 x float> %T1) nounwind readnone {
entry:
	%tmp7 = shufflevector <4 x float> %a, <4 x float> undef, <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3 >		; <<8 x float>> [#uses=1]
	%sub = fsub <8 x float> %T1, %T0		; <<8 x float>> [#uses=1]
	%mul = fmul <8 x float> %sub, %tmp7		; <<8 x float>> [#uses=1]
	%add = fadd <8 x float> %mul, %T0		; <<8 x float>> [#uses=1]
	ret <8 x float> %add
}
