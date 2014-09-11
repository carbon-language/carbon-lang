; RUN: llc < %s -mtriple=sparc-unknown-openbsd -disable-fp-elim | FileCheck -check-prefix=CHECK-FP-LABEL %s

define void @func() {
entry:
  unreachable
}
; CHECK-FP-LABEL:      {{_?}}func:
; CHECK-FP-LABEL: nop {{[;!]}} avoids zero-length function
