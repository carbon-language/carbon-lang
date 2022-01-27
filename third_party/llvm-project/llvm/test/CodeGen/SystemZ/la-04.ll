; Test blockaddress.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Do some arbitrary work and return the address of the following label.
define i8 *@f1(i8 *%addr) {
; CHECK-LABEL: f1:
; CHECK: mvi 0(%r2), 1
; CHECK: [[LABEL:\.L.*]]:
; CHECK: larl %r2, [[LABEL]]
; CHECK: br %r14
entry:
  store i8 1, i8 *%addr
  br label %b.lab

b.lab:
  ret i8 *blockaddress(@f1, %b.lab)
}
