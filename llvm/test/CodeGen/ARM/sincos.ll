; RUN: llc < %s -mtriple=armv7-apple-ios6 -mcpu=cortex-a8 | FileCheck %s --check-prefix=NOOPT
; RUN: llc < %s -mtriple=armv7-apple-ios7 -mcpu=cortex-a8 | FileCheck %s --check-prefix=SINCOS

; Combine sin / cos into a single call.
; rdar://12856873

define float @test1(float %x) nounwind {
entry:
; SINCOS-LABEL: test1:
; SINCOS: bl ___sincosf_stret

; NOOPT-LABEL: test1:
; NOOPT: bl _sinf
; NOOPT: bl _cosf
  %call = tail call float @sinf(float %x) nounwind readnone
  %call1 = tail call float @cosf(float %x) nounwind readnone
  %add = fadd float %call, %call1
  ret float %add
}

define double @test2(double %x) nounwind {
entry:
; SINCOS-LABEL: test2:
; SINCOS: bl ___sincos_stret

; NOOPT-LABEL: test2:
; NOOPT: bl _sin
; NOOPT: bl _cos
  %call = tail call double @sin(double %x) nounwind readnone
  %call1 = tail call double @cos(double %x) nounwind readnone
  %add = fadd double %call, %call1
  ret double %add
}

declare float  @sinf(float) readonly
declare double @sin(double) readonly
declare float @cosf(float) readonly
declare double @cos(double) readonly
