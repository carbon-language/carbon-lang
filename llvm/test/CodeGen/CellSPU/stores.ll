; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep {stqd.*0(\$3)}      %t1.s | count 4
; RUN: grep {stqd.*16(\$3)}     %t1.s | count 4
; RUN: grep 16256               %t1.s | count 2
; RUN: grep 16384               %t1.s | count 1
; RUN: grep {shli.*, 4}         %t1.s | count 4
; RUN: grep stqx                %t1.s | count 4

; ModuleID = 'stores.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define void @store_v16i8_1(<16 x i8>* %a) nounwind {
entry:
	store <16 x i8> < i8 1, i8 2, i8 1, i8 1, i8 1, i8 2, i8 1, i8 1, i8 1, i8 2, i8 1, i8 1, i8 1, i8 2, i8 1, i8 1 >, <16 x i8>* %a
	ret void
}

define void @store_v16i8_2(<16 x i8>* %a) nounwind {
entry:
	%arrayidx = getelementptr <16 x i8>* %a, i32 1
	store <16 x i8> < i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2 >, <16 x i8>* %arrayidx
	ret void
}

define void @store_v16i8_3(<16 x i8>* %a, i32 %i) nounwind {
entry:
        %arrayidx = getelementptr <16 x i8>* %a, i32 %i
	store <16 x i8> < i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1 >, <16 x i8>* %arrayidx
        ret void
}

define void @store_v8i16_1(<8 x i16>* %a) nounwind {
entry:
	store <8 x i16> < i16 1, i16 2, i16 1, i16 1, i16 1, i16 2, i16 1, i16 1 >, <8 x i16>* %a
	ret void
}

define void @store_v8i16_2(<8 x i16>* %a) nounwind {
entry:
	%arrayidx = getelementptr <8 x i16>* %a, i16 1
	store <8 x i16> < i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2 >, <8 x i16>* %arrayidx
	ret void
}

define void @store_v8i16_3(<8 x i16>* %a, i32 %i) nounwind {
entry:
        %arrayidx = getelementptr <8 x i16>* %a, i32 %i
	store <8 x i16> < i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1 >, <8 x i16>* %arrayidx
        ret void
}

define void @store_v4i32_1(<4 x i32>* %a) nounwind {
entry:
	store <4 x i32> < i32 1, i32 2, i32 1, i32 1 >, <4 x i32>* %a
	ret void
}

define void @store_v4i32_2(<4 x i32>* %a) nounwind {
entry:
	%arrayidx = getelementptr <4 x i32>* %a, i32 1
	store <4 x i32> < i32 2, i32 2, i32 2, i32 2 >, <4 x i32>* %arrayidx
	ret void
}

define void @store_v4i32_3(<4 x i32>* %a, i32 %i) nounwind {
entry:
        %arrayidx = getelementptr <4 x i32>* %a, i32 %i
        store <4 x i32> < i32 1, i32 1, i32 1, i32 1 >, <4 x i32>* %arrayidx
        ret void
}

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

define void @store_v4f32_3(<4 x float>* %a, i32 %i) nounwind {
entry:
        %arrayidx = getelementptr <4 x float>* %a, i32 %i
        store <4 x float> < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >, <4 x float>* %arrayidx
        ret void
}
