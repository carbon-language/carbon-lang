; RUN: llc -mtriple armv7 -O0 -o - < %s | FileCheck %s -check-prefix CHECK-TAIL
; RUN: llc -mtriple armv7 -O0 -disable-tail-calls -o - < %s \
; RUN:   | FileCheck %s -check-prefix CHECK-NO-TAIL

declare i32 @callee(i32 %i)

define i32 @caller(i32 %i) {
entry:
  %r = tail call i32 @callee(i32 %i)
  ret i32 %r
}

; CHECK-TAIL-LABEL: caller
; CHECK-TAIL: b callee

; CHECK-NO-TAIL-LABEL: caller
; CHECK-NO-TAIL: push {lr}
; CHECK-NO-TAIL: bl callee
; CHECK-NO-TAIL: pop {lr}
; CHECK-NO-TAIL: bx lr

