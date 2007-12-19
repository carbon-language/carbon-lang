; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep fceq  %t1.s | count 1 &&
; RUN: grep fcmeq %t1.s | count 1
;
; This file includes standard floating point arithmetic instructions

declare double @fabs(double)
declare float @fabsf(float)

define i1 @fcmp_eq(float %arg1, float %arg2) {
	%A = fcmp oeq float %arg1,  %arg2 	; <float> [#uses=1]
	ret i1 %A
}

define i1 @fcmp_mag_eq(float %arg1, float %arg2) {
	%A = call float @fabsf(float %arg1)	; <float> [#uses=1]
	%B = call float @fabsf(float %arg2)	; <float> [#uses=1]
	%C = fcmp oeq float %A,  %B	; <float> [#uses=1]
	ret i1 %C
}
