; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep frest    %t1.s | count 2 &&
; RUN: grep fi       %t1.s | count 2 &&
; RUN: grep fm       %t1.s | count 4 &&
; RUN: grep fma      %t1.s | count 2 &&
; RUN: grep fnms     %t1.s | count 2
;
; This file includes standard floating point arithmetic instructions

define float @fdiv32(float %arg1, float %arg2) {
	%A = fdiv float %arg1,  %arg2
	ret float %A
}

define <4 x float> @fdiv_v4f32(<4 x float> %arg1, <4 x float> %arg2) {
	%A = fdiv <4 x float> %arg1,  %arg2
 	ret <4 x float> %A
}
