; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic conversion operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: $wrap_i64_i32
; CHECK-NEXT: (param i64) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (wrap_i64 @0))
; CHECK-NEXT: (return @1)
define i32 @wrap_i64_i32(i64 %x) {
  %a = trunc i64 %x to i32
  ret i32 %a
}
