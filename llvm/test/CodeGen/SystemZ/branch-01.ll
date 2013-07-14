; Test a simple unconditional jump.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(i8 *%dest) {
; CHECK-LABEL: f1:
; CHECK: .L[[LABEL:.*]]:
; CHECK: mvi 0(%r2), 1
; CHECK: j .L[[LABEL]]
  br label %loop
loop:
  store volatile i8 1, i8 *%dest
  br label %loop
}
