; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test various types and operators that need to be legalized.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: shl_i3:
; CHECK: i32.const   $push0=, 7
; CHECK: i32.and     $push1=, $1, $pop0
; CHECK: i32.shl     $push2=, $0, $pop1
define i3 @shl_i3(i3 %a, i3 %b, i3* %p) {
  %t = shl i3 %a, %b
  ret i3 %t
}

; CHECK-LABEL: shl_i53:
; CHECK: i64.const   $push0=, 9007199254740991
; CHECK: i64.and     $push1=, $1, $pop0
; CHECK: i64.shl     $push2=, $0, $pop1
define i53 @shl_i53(i53 %a, i53 %b, i53* %p) {
  %t = shl i53 %a, %b
  ret i53 %t
}
