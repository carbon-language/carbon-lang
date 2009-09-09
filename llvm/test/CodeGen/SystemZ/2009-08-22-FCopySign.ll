; RUN: llc < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:16:16-f128:128:128"
target triple = "s390x-ibm-linux-gnu"

define double @foo(double %a, double %b) nounwind {
entry:
; CHECK: cpsdr %f0, %f2, %f0
  %0 = tail call double @copysign(double %a, double %b) nounwind readnone
  ret double %0
}

define float @bar(float %a, float %b) nounwind {
entry:
; CHECK: cpsdr %f0, %f2, %f0
  %0 = tail call float @copysignf(float %a, float %b) nounwind readnone
  ret float %0
}


declare double @copysign(double, double) nounwind readnone
declare float @copysignf(float, float) nounwind readnone
