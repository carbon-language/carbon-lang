; RUN: llc < %s -march=x86 -mattr=+sse2 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define float @foo(float %x) nounwind readnone ssp {
entry:
; CHECK-NOT: cvtss2sd
; CHECK-NOT: sqrtsd
; CHECK-NOT: cvtsd2ss
; CHECK: sqrtss
  %conv = fpext float %x to double                ; <double> [#uses=1]
  %call = tail call double @sqrt(double %conv) nounwind ; <double> [#uses=1]
  %conv1 = fptrunc double %call to float          ; <float> [#uses=1]
  ret float %conv1
}

declare double @sqrt(double) readnone
