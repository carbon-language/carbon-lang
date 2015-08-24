; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic call operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @i32_nullary()
declare i32 @i32_unary(i32)
declare i64 @i64_nullary()
declare float @float_nullary()
declare double @double_nullary()

; CHECK-LABEL: call_i32_nullary:
; CHECK-NEXT: (setlocal @0 (global $i32_nullary))
; CHECK-NEXT: (setlocal @1 (call @0))
; CHECK-NEXT: (return @1)
define i32 @call_i32_nullary() {
  %r = call i32 @i32_nullary()
  ret i32 %r
}

; CHECK-LABEL: call_i64_nullary:
; CHECK-NEXT: (setlocal @0 (global $i64_nullary))
; CHECK-NEXT: (setlocal @1 (call @0))
; CHECK-NEXT: (return @1)
define i64 @call_i64_nullary() {
  %r = call i64 @i64_nullary()
  ret i64 %r
}

; CHECK-LABEL: call_float_nullary:
; CHECK-NEXT: (setlocal @0 (global $float_nullary))
; CHECK-NEXT: (setlocal @1 (call @0))
; CHECK-NEXT: (return @1)
define float @call_float_nullary() {
  %r = call float @float_nullary()
  ret float %r
}

; CHECK-LABEL: call_double_nullary:
; CHECK-NEXT: (setlocal @0 (global $double_nullary))
; CHECK-NEXT: (setlocal @1 (call @0))
; CHECK-NEXT: (return @1)
define double @call_double_nullary() {
  %r = call double @double_nullary()
  ret double %r
}

; CHECK-LABEL: call_i32_unary:
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (global $i32_unary))
; CHECK-NEXT: (setlocal @2 (call @1 @0))
; CHECK-NEXT: (return @2)
define i32 @call_i32_unary(i32 %a) {
  %r = call i32 @i32_unary(i32 %a)
  ret i32 %r
}

; FIXME test the following:
;  - Functions without return.
;  - More argument combinations.
;  - Tail call.
;  - Interesting returns (struct, multiple).
;  - Vararg.
