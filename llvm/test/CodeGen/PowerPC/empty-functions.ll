; RUN: llc < %s -mtriple=powerpc-apple-darwin | FileCheck -check-prefix=CHECK-NO-FP %s
; RUN: llc < %s -mtriple=powerpc-apple-darwin -disable-fp-elim | FileCheck -check-prefix=CHECK-FP %s
; RUN: llc < %s -mtriple=powerpc-netbsd -disable-fp-elim | FileCheck -check-prefix=CHECK-FP %s

define void @func() {
entry:
  unreachable
}
; CHECK-NO-FP:     _func:
; CHECK-NO-FP:     nop

; CHECK-FP:      {{_?}}func:
; CHECK-FP: nop {{[;#]}} avoids zero-length function
