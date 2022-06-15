; Verify that the driver can consume LLVM IR files.

; REQUIRES: aarch64-registered-target

;-------------
; RUN COMMANDS
;-------------
; RUN: %flang_fc1 -S -triple aarch64-unknown-linux-gnu %s -o - | FileCheck %s
; RUN: %flang -S -target aarch64-unknown-linux-gnu %s -o - | FileCheck %s

;----------------
; EXPECTED OUTPUT
;----------------
; CHECK-LABEL: foo:
; CHECK: ret

;------
; INPUT
;------
define void @foo() {
  ret void
}
