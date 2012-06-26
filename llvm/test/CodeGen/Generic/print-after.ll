; RUN: not llc --help-hidden |& FileCheck %s

; CHECK: -print-after
; CHECK-NOT: -print-after-all
; CHECK: =simple-register-coalescing
; CHECK: -print-after-all
