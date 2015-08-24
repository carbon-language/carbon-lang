; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic call operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @nullary()

; CHECK-LABEL: call_nullary:
; CHECK-NEXT: (call @foo)
; CHECK-NEXT: (return)
define void @call_nullary() {
  call void @nullary()
  ret void
}


; tail call
; multiple args
; interesting returns (int, float, struct, multiple)
; vararg
