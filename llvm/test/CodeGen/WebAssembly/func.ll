; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic functions assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: (func $f0{{$}}
; CHECK: (return){{$}}
; CHECK-NEXT: ) ;; end func $f0{{$}}
define void @f0() {
  ret void
}

; CHECK-LABEL: (func $f1{{$}}
; CHECK-NEXT: (result i32){{$}}
; CHECK-NEXT: (setlocal @0 (immediate 0)){{$}}
; CHECK-NEXT: (return @0){{$}}
; CHECK-NEXT: ) ;; end func $f1{{$}}
define i32 @f1() {
  ret i32 0
}

; CHECK-LABEL: (func $f2{{$}}
; CHECK-NEXT: (param i32) (param f32) (result i32){{$}}
; CHECK-NEXT: (setlocal @0 (immediate 0)){{$}}
; CHECK-NEXT: (return @0){{$}}
; CHECK-NEXT: ) ;; end func $f2{{$}}
define i32 @f2(i32 %p1, float %p2) {
  ret i32 0
}

; CHECK-LABEL: (func $f3{{$}}
; CHECK-NEXT: (param i32) (param f32){{$}}
; CHECK-NEXT: (return){{$}}
; CHECK-NEXT: ) ;; end func $f3{{$}}
define void @f3(i32 %p1, float %p2) {
  ret void
}
