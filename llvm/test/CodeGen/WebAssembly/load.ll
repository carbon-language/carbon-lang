; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic loads are assembled properly.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ldi32:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @1, pop{{$}}
; CHECK-NEXT: load_i32 @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: return @2{{$}}
define i32 @ldi32(i32 *%p) {
  %v = load i32, i32* %p
  ret i32 %v
}

; CHECK-LABEL: ldi64:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @1, pop{{$}}
; CHECK-NEXT: load_i64 @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: return @2{{$}}
define i64 @ldi64(i64 *%p) {
  %v = load i64, i64* %p
  ret i64 %v
}

; CHECK-LABEL: ldf32:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result f32{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @1, pop{{$}}
; CHECK-NEXT: load_f32 @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: return @2{{$}}
define float @ldf32(float *%p) {
  %v = load float, float* %p
  ret float %v
}

; CHECK-LABEL: ldf64:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result f64{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @1, pop{{$}}
; CHECK-NEXT: load_f64 @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: return @2{{$}}
define double @ldf64(double *%p) {
  %v = load double, double* %p
  ret double %v
}
