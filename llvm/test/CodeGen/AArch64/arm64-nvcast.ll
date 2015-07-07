; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s

; CHECK-LABEL: _test:
; CHECK:  fmov.2d v0, #2.00000000
; CHECK:  str  q0, [sp]
; CHECK:  mov  x8, sp
; CHECK:  ldr s0, [x8, w1, sxtw #2]
; CHECK:  str  s0, [x0]

define void @test(float * %p1, i32 %v1) {
entry:
  %v2 = extractelement <3 x float> <float 0.000000e+00, float 2.000000e+00, float 0.000000e+00>, i32 %v1
  store float %v2, float* %p1, align 4
  ret void
}
