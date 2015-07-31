; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Make sure that argument offsets are correct even if some arguments are unused.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: unused_first:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (RETURN @0)
define i32 @unused_first(i32 %x, i32 %y) {
  ret i32 %y
}

; CHECK-LABEL: unused_second:
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (RETURN @0)
define i32 @unused_second(i32 %x, i32 %y) {
  ret i32 %x
}
