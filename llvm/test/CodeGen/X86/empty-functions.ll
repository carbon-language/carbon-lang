; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck -check-prefix=CHECK-NO-FP %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -disable-fp-elim | FileCheck -check-prefix=CHECK-FP %s

define void @func() {
entry:
  unreachable
}
; CHECK-NO-FP:     _func:
; CHECK-NO-FP-NOT: movq %rsp, %rbp
; CHECK-NO-FP:     nop

; CHECK-FP:      _func:
; CHECK-FP:      movq %rsp, %rbp
; CHECK-FP-NEXT: Ltmp1:
; CHECK-FP:      nop
