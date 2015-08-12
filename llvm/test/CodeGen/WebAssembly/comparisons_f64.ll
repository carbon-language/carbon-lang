; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit floating-point comparison operations assemble as
; expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: ord_f64:
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (argument 1))
; CHECK-NEXT: (setlocal @2 (eq @1 @1))
; CHECK-NEXT: (setlocal @3 (eq @0 @0))
; CHECK-NEXT: (setlocal @4 (and @3 @2))
; CHECK-NEXT: (return @4)
define i32 @ord_f64(double %x, double %y) {
  %a = fcmp ord double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uno_f64:
; CHECK-NEXT: (setlocal @0 (argument 0))
; CHECK-NEXT: (setlocal @1 (argument 1))
; CHECK-NEXT: (setlocal @2 (ne @1 @1))
; CHECK-NEXT: (setlocal @3 (ne @0 @0))
; CHECK-NEXT: (setlocal @4 (ior @3 @2))
; CHECK-NEXT: (return @4)
define i32 @uno_f64(double %x, double %y) {
  %a = fcmp uno double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: oeq_f64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (eq @1 @0))
; CHECK-NEXT: (return @2)
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

; Expanded comparisons, which also check for NaN.

; CHECK-LABEL: ueq_f64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (eq @1 @0))
; CHECK-NEXT: (setlocal @3 (ne @0 @0))
; CHECK-NEXT: (setlocal @4 (ne @1 @1))
; CHECK-NEXT: (setlocal @5 (ior @4 @3))
; CHECK-NEXT: (setlocal @6 (ior @2 @5))
; CHECK-NEXT: (return @6)
define i32 @ueq_f64(double %x, double %y) {
  %a = fcmp ueq double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: one_f64:
; CHECK: (setlocal @2 (ne @1 @0))
define i32 @one_f64(double %x, double %y) {
  %a = fcmp one double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ult_f64:
; CHECK: (setlocal @2 (lt @1 @0))
define i32 @ult_f64(double %x, double %y) {
  %a = fcmp ult double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ule_f64:
; CHECK: (setlocal @2 (le @1 @0))
define i32 @ule_f64(double %x, double %y) {
  %a = fcmp ule double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: ugt_f64:
; CHECK: (setlocal @2 (gt @1 @0))
define i32 @ugt_f64(double %x, double %y) {
  %a = fcmp ugt double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}

; CHECK-LABEL: uge_f64:
; CHECK: (setlocal @2 (ge @1 @0))
define i32 @uge_f64(double %x, double %y) {
  %a = fcmp uge double %x, %y
  %b = zext i1 %a to i32
  ret i32 %b
}
