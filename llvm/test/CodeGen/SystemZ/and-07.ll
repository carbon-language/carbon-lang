; Test the three-operand forms of AND.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check NRK.
define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f1:
; CHECK: nrk %r2, %r3, %r4
; CHECK: br %r14
  %and = and i32 %b, %c
  ret i32 %and
}

; Check that we can still use NR in obvious cases.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: nr %r2, %r3
; CHECK: br %r14
  %and = and i32 %a, %b
  ret i32 %and
}
