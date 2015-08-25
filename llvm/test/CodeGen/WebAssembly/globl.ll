; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that the `.globl` directive is commented out.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-NOT: globl
; CHECK: ;; .globl foo
; CHECK-NOT: globl
; CHECK-LABEL: foo:
define void @foo() {
  ret void
}
