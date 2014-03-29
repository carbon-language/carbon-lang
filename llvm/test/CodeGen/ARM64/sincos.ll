; RUN: llc < %s -mtriple=arm64-apple-ios7 | FileCheck %s

; Combine sin / cos into a single call.
; rdar://12856873

define float @test1(float %x) nounwind {
entry:
; CHECK-LABEL: test1:
; CHECK: bl ___sincosf_stret
; CHECK: fadd s0, s0, s1
  %call = tail call float @sinf(float %x) nounwind readnone
  %call1 = tail call float @cosf(float %x) nounwind readnone
  %add = fadd float %call, %call1
  ret float %add
}

define double @test2(double %x) nounwind {
entry:
; CHECK-LABEL: test2:
; CHECK: bl ___sincos_stret
; CHECK: fadd d0, d0, d1
  %call = tail call double @sin(double %x) nounwind readnone
  %call1 = tail call double @cos(double %x) nounwind readnone
  %add = fadd double %call, %call1
  ret double %add
}

declare float  @sinf(float) readonly
declare double @sin(double) readonly
declare float @cosf(float) readonly
declare double @cos(double) readonly
