; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep {stqd.*0(\$3)}   %t1.s | count 1
; RUN: grep {stqd.*16(\$3)}  %t1.s | count 1
; RUN: grep 16256            %t1.s | count 1
; RUN: grep 16384            %t1.s | count 1

; ModuleID = 'stores.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define void @store_v4f32_1(<4 x float>* %a) nounwind {
entry:
	store <4 x float> < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >, <4 x float>* %a
	ret void
}

define void @store_v4f32_2(<4 x float>* %a) nounwind {
entry:
	%arrayidx = getelementptr <4 x float>* %a, i32 1
	store <4 x float> < float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00 >, <4 x float>* %arrayidx
	ret void
}
