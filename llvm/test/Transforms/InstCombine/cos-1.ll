; Test that the cos library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s -check-prefix=NO-FLOAT-SHRINK
; RUN: opt < %s -instcombine -enable-double-float-shrink -S | FileCheck %s -check-prefix=DO-FLOAT-SHRINK

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare double @cos(double)

; Check cos(-x) -> cos(x);

define double @test_simplify1(double %d) {
; NO-FLOAT-SHRINK: @test_simplify1
  %neg = fsub double -0.000000e+00, %d
  %cos = call double @cos(double %neg)
; NO-FLOAT-SHRINK: call double @cos(double %d)
  ret double %cos
}

define float @test_simplify2(float %f) {
; DO-FLOAT-SHRINK: @test_simplify2
  %conv1 = fpext float %f to double
  %neg = fsub double -0.000000e+00, %conv1
  %cos = call double @cos(double %neg)
  %conv2 = fptrunc double %cos to float
; DO-FLOAT-SHRINK: call float @cosf(float %f)
  ret float %conv2
}

define float @test_simplify3(float %f) {
; NO-FLOAT-SHRINK: @test_simplify3
  %conv1 = fpext float %f to double
  %neg = fsub double -0.000000e+00, %conv1
  %cos = call double @cos(double %neg)
; NO-FLOAT-SHRINK: call double @cos(double %conv1)
  %conv2 = fptrunc double %cos to float
  ret float %conv2
}
