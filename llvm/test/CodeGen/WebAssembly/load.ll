; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic loads are assembled properly.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: (func $ldi32
; CHECK-NEXT: (param i32) (result i32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (load_i32 @0))
; CHECK-NEXT: (return @1)
define i32 @ldi32(i32 *%p) {
  %v = load i32, i32* %p
  ret i32 %v
}

; CHECK-LABEL: (func $ldi64
; CHECK-NEXT: (param i32) (result i64)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (load_i64 @0))
; CHECK-NEXT: (return @1)
define i64 @ldi64(i64 *%p) {
  %v = load i64, i64* %p
  ret i64 %v
}

; CHECK-LABEL: (func $ldf32
; CHECK-NEXT: (param i32) (result f32)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (load_f32 @0))
; CHECK-NEXT: (return @1)
define float @ldf32(float *%p) {
  %v = load float, float* %p
  ret float %v
}

; CHECK-LABEL: (func $ldf64
; CHECK-NEXT: (param i32) (result f64)
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (load_f64 @0))
; CHECK-NEXT: (return @1)
define double @ldf64(double *%p) {
  %v = load double, double* %p
  ret double %v
}
