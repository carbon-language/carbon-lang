; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit floating-point comparison operations assemble as
; expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ord_f32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: eq @3, @3{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: eq @2, @2{{$}}
; CHECK-NEXT: set_local @5, pop{{$}}
; CHECK-NEXT: and @5, @4{{$}}
; CHECK-NEXT: set_local @6, pop{{$}}
; CHECK-NEXT: return @6{{$}}
define i32 @ord_f32(float %x, float %y) {
  %a = fcmp ord float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uno_f32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: ne @3, @3{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: ne @2, @2{{$}}
; CHECK-NEXT: set_local @5, pop{{$}}
; CHECK-NEXT: ior @5, @4{{$}}
; CHECK-NEXT: set_local @6, pop{{$}}
; CHECK-NEXT: return @6{{$}}
define i32 @uno_f32(float %x, float %y) {
  %a = fcmp uno float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oeq_f32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: eq @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: return @4{{$}}
define i32 @oeq_f32(float %x, float %y) {
  %a = fcmp oeq float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: une_f32:
; CHECK: ne @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @une_f32(float %x, float %y) {
  %a = fcmp une float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f32:
; CHECK: lt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @olt_f32(float %x, float %y) {
  %a = fcmp olt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ole_f32:
; CHECK: le @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ole_f32(float %x, float %y) {
  %a = fcmp ole float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ogt_f32:
; CHECK: gt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ogt_f32(float %x, float %y) {
  %a = fcmp ogt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oge_f32:
; CHECK: ge @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @oge_f32(float %x, float %y) {
  %a = fcmp oge float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; Expanded comparisons, which also check for NaN.

; CHECK-LABEL: ueq_f32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: @1{{$}}
; CHECK-NEXT: set_local @2, pop{{$}}
; CHECK-NEXT: @0{{$}}
; CHECK-NEXT: set_local @3, pop{{$}}
; CHECK-NEXT: eq @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
; CHECK-NEXT: ne @2, @2{{$}}
; CHECK-NEXT: set_local @5, pop{{$}}
; CHECK-NEXT: ne @3, @3{{$}}
; CHECK-NEXT: set_local @6, pop{{$}}
; CHECK-NEXT: ior @6, @5{{$}}
; CHECK-NEXT: set_local @7, pop{{$}}
; CHECK-NEXT: ior @4, @7{{$}}
; CHECK-NEXT: set_local @8, pop{{$}}
; CHECK-NEXT: return @8{{$}}
define i32 @ueq_f32(float %x, float %y) {
  %a = fcmp ueq float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: one_f32:
; CHECK: ne @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @one_f32(float %x, float %y) {
  %a = fcmp one float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_f32:
; CHECK: lt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ult_f32(float %x, float %y) {
  %a = fcmp ult float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_f32:
; CHECK: le @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ule_f32(float %x, float %y) {
  %a = fcmp ule float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_f32:
; CHECK: gt @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @ugt_f32(float %x, float %y) {
  %a = fcmp ugt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_f32:
; CHECK: ge @3, @2{{$}}
; CHECK-NEXT: set_local @4, pop{{$}}
define i32 @uge_f32(float %x, float %y) {
  %a = fcmp uge float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
