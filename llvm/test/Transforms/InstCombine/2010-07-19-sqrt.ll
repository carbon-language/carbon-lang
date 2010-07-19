; RUN: opt -S -instcombine %s | FileCheck %s

define float @foo(float %x) nounwind readnone ssp {
entry:
; CHECK-NOT: fpext
; CHECK-NOT: sqrt(
; CHECK: sqrtf(
; CHECK-NOT: fptrunc
  %conv = fpext float %x to double                ; <double> [#uses=1]
  %call = tail call double @sqrt(double %conv) nounwind ; <double> [#uses=1]
  %conv1 = fptrunc double %call to float          ; <float> [#uses=1]
; CHECK: ret float
  ret float %conv1
}

declare double @sqrt(double) readnone
