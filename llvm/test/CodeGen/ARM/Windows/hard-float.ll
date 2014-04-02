; RUN: llc -mtriple=thumbv7-windows-itanium -mcpu=cortex-a9 -o - %s | FileCheck %s

define float @function(float %f, float %g) nounwind {
entry:
  %h = fadd float %f, %g
  ret float %h
}

; CHECK: vadd.f32 s0, s0, s1

