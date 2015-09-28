; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic call operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare i32 @i32_nullary()
declare i32 @i32_unary(i32)
declare i32 @i32_binary(i32, i32)
declare i64 @i64_nullary()
declare float @float_nullary()
declare double @double_nullary()
declare void @void_nullary()

; CHECK-LABEL: (func $call_i32_nullary
; CHECK-NEXT: (result i32)
; CHECK-NEXT: (setlocal @0 (call $i32_nullary))
; CHECK-NEXT: (return @0)
define i32 @call_i32_nullary() {
  %r = call i32 @i32_nullary()
  ret i32 %r
}

; CHECK-LABEL: (func $call_i64_nullary
; CHECK-NEXT: (result i64)
; CHECK-NEXT: (setlocal @0 (call $i64_nullary))
; CHECK-NEXT: (return @0)
define i64 @call_i64_nullary() {
  %r = call i64 @i64_nullary()
  ret i64 %r
}

; CHECK-LABEL: (func $call_float_nullary
; CHECK-NEXT: (result f32)
; CHECK-NEXT: (setlocal @0 (call $float_nullary))
; CHECK-NEXT: (return @0)
define float @call_float_nullary() {
  %r = call float @float_nullary()
  ret float %r
}

; CHECK-LABEL: (func $call_double_nullary
; CHECK-NEXT: (result f64)
; CHECK-NEXT: (setlocal @0 (call $double_nullary))
; CHECK-NEXT: (return @0)
define double @call_double_nullary() {
  %r = call double @double_nullary()
  ret double %r
}

; CHECK-LABEL: (func $call_void_nullary
; CHECK-NEXT: (call $void_nullary)
; CHECK-NEXT: (return)
define void @call_void_nullary() {
  call void @void_nullary()
  ret void
}

; CHECK-LABEL: (func $call_i32_unary
; CHECK-NEXT: (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (call $i32_unary @0))
; CHECK-NEXT: (return @1)
define i32 @call_i32_unary(i32 %a) {
  %r = call i32 @i32_unary(i32 %a)
  ret i32 %r
}

; CHECK-LABEL: (func $call_i32_binary
; CHECK-NEXT: (param i32) (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (call $i32_binary @1 @0))
; CHECK-NEXT: (return @2)
define i32 @call_i32_binary(i32 %a, i32 %b) {
  %r = call i32 @i32_binary(i32 %a, i32 %b)
  ret i32 %r
}

; CHECK-LABEL: (func $call_indirect_void
; CHECK-NEXT: (param i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (call_indirect @0)
; CHECK-NEXT: (return)
define void @call_indirect_void(void ()* %callee) {
  call void %callee()
  ret void
}

; CHECK-LABEL: (func $call_indirect_i32
; CHECK-NEXT: (param i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (call_indirect @0))
; CHECK-NEXT: (return @1)
define i32 @call_indirect_i32(i32 ()* %callee) {
  %t = call i32 %callee()
  ret i32 %t
}

; FIXME test the following:
;  - Functions without return.
;  - More argument combinations.
;  - Tail call.
;  - Interesting returns (struct, multiple).
;  - Vararg.
