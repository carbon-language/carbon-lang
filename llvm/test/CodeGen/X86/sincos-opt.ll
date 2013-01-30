; RUN: llc < %s -mtriple=x86_64-apple-macosx10.9.0 -mcpu=core2 | FileCheck %s --check-prefix=SINCOS
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.8.0 -mcpu=core2 | FileCheck %s --check-prefix=NOOPT

; Combine sin / cos into a single call.
; rdar://13087969

define float @test1(float %x) nounwind {
entry:
; SINCOS: test1:
; SINCOS: callq ___sincosf_stret
; SINCOS: addss %xmm1, %xmm0

; NOOPT: test1
; NOOPT: callq _cosf
; NOOPT: callq _sinf
  %call = tail call float @sinf(float %x) nounwind readnone
  %call1 = tail call float @cosf(float %x) nounwind readnone
  %add = fadd float %call, %call1
  ret float %add
}

define double @test2(double %x) nounwind {
entry:
; SINCOS: test2:
; SINCOS: callq ___sincos_stret
; SINCOS: addsd %xmm1, %xmm0

; NOOPT: test2
; NOOPT: callq _cos
; NOOPT: callq _sin
  %call = tail call double @sin(double %x) nounwind readnone
  %call1 = tail call double @cos(double %x) nounwind readnone
  %add = fadd double %call, %call1
  ret double %add
}

declare float  @sinf(float) readonly
declare double @sin(double) readonly
declare float @cosf(float) readonly
declare double @cos(double) readonly
