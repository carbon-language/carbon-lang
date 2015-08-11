; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit floating-point comparison operations assemble as
; expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; FIXME: add ord and uno tests.

; CHECK-LABEL: oeq_f64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (eq @1 @0))
; CHECK-NEXT: (setlocal @3 (immediate 1))
; CHECK-NEXT: (setlocal @4 (and @2 @3))
; CHECK-NEXT: (return @4)
define i32 @oeq_f64(double %x, double %y) {
  %a = fcmp oeq double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: une_f64:
; CHECK: (setlocal @2 (ne @1 @0))
define i32 @une_f64(double %x, double %y) {
  %a = fcmp une double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: olt_f64:
; CHECK: (setlocal @2 (lt @1 @0))
define i32 @olt_f64(double %x, double %y) {
  %a = fcmp olt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ole_f64:
; CHECK: (setlocal @2 (le @1 @0))
define i32 @ole_f64(double %x, double %y) {
  %a = fcmp ole double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ogt_f64:
; CHECK: (setlocal @2 (gt @1 @0))
define i32 @ogt_f64(double %x, double %y) {
  %a = fcmp ogt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oge_f64:
; CHECK: (setlocal @2 (ge @1 @0))
define i32 @oge_f64(double %x, double %y) {
  %a = fcmp oge double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; FIXME test other FP comparisons: ueq, one, ult, ule, ugt, uge. They currently
; are broken and failt to match.
