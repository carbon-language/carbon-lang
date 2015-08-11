; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit floating-point comparison operations assemble as
; expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; FIXME: add ord and uno tests.

; CHECK-LABEL: oeq_f32:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (eq @1 @0))
; CHECK-NEXT: (setlocal @3 (immediate 1))
; CHECK-NEXT: (setlocal @4 (and @2 @3))
; CHECK-NEXT: (return @4)
define i32 @oeq_f32(float %x, float %y) {
  %a = fcmp oeq float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: une_f32:
; CHECK: (setlocal @2 (ne @1 @0))
define i32 @une_f32(float %x, float %y) {
  %a = fcmp une float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f32:
; CHECK: (setlocal @2 (lt @1 @0))
define i32 @olt_f32(float %x, float %y) {
  %a = fcmp olt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ole_f32:
; CHECK: (setlocal @2 (le @1 @0))
define i32 @ole_f32(float %x, float %y) {
  %a = fcmp ole float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ogt_f32:
; CHECK: (setlocal @2 (gt @1 @0))
define i32 @ogt_f32(float %x, float %y) {
  %a = fcmp ogt float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oge_f32:
; CHECK: (setlocal @2 (ge @1 @0))
define i32 @oge_f32(float %x, float %y) {
  %a = fcmp oge float %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; FIXME test other FP comparisons: ueq, one, ult, ule, ugt, uge. They currently
; are broken and failt to match.
