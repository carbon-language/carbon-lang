; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep fceq  %t1.s | count 1
; RUN: grep fcmeq %t1.s | count 1
;
; This file includes standard floating point arithmetic instructions
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

declare double @fabs(double)
declare float @fabsf(float)

define i1 @fcmp_eq(float %arg1, float %arg2) {
        %A = fcmp oeq float %arg1,  %arg2       ; <float> [#uses=1]
        ret i1 %A
}

define i1 @fcmp_mag_eq(float %arg1, float %arg2) {
        %A = call float @fabsf(float %arg1)     ; <float> [#uses=1]
        %B = call float @fabsf(float %arg2)     ; <float> [#uses=1]
        %C = fcmp oeq float %A,  %B     ; <float> [#uses=1]
        ret i1 %C
}
