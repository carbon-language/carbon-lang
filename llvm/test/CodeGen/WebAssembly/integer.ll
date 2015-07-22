; RUN: llc < %s -asm-verbose=false | FileCheck %s

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: add32:
define i32 @add32(i32 %x, i32 %y) {
  %a = add i32 %x, %y
  ret i32 %a
}
