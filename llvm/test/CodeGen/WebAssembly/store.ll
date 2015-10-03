; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic stores are assembled properly.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: (func $sti32
; CHECK-NEXT: (param i32) (param i32)
; CHECK-NEXT: (set_local @0 (argument 1))
; CHECK-NEXT: (set_local @1 (argument 0))
; CHECK-NEXT: (store_i32 @1 @0)
; CHECK-NEXT: (return)
define void @sti32(i32 *%p, i32 %v) {
  store i32 %v, i32* %p
  ret void
}

; CHECK-LABEL: (func $sti64
; CHECK-NEXT: (param i32) (param i64)
; CHECK-NEXT: (set_local @0 (argument 1))
; CHECK-NEXT: (set_local @1 (argument 0))
; CHECK-NEXT: (store_i64 @1 @0)
; CHECK-NEXT: (return)
define void @sti64(i64 *%p, i64 %v) {
  store i64 %v, i64* %p
  ret void
}

; CHECK-LABEL: (func $stf32
; CHECK-NEXT: (param i32) (param f32)
; CHECK-NEXT: (set_local @0 (argument 1))
; CHECK-NEXT: (set_local @1 (argument 0))
; CHECK-NEXT: (store_f32 @1 @0)
; CHECK-NEXT: (return)
define void @stf32(float *%p, float %v) {
  store float %v, float* %p
  ret void
}

; CHECK-LABEL: (func $stf64
; CHECK-NEXT: (param i32) (param f64)
; CHECK-NEXT: (set_local @0 (argument 1))
; CHECK-NEXT: (set_local @1 (argument 0))
; CHECK-NEXT: (store_f64 @1 @0)
; CHECK-NEXT: (return)
define void @stf64(double *%p, double %v) {
  store double %v, double* %p
  ret void
}
