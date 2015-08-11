; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit floating-point operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare double @llvm.fabs.f64(double)
declare double @llvm.copysign.f64(double, double)
declare double @llvm.sqrt.f64(double)

; CHECK-LABEL: fadd64:
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (fadd @1 @0))
; CHECK-NEXT: (return @2)
define double @fadd64(double %x, double %y) {
  %a = fadd double %x, %y
  ret double %a
}

; CHECK-LABEL: fsub64:
; CHECK: (setlocal @2 (fsub @1 @0))
define double @fsub64(double %x, double %y) {
  %a = fsub double %x, %y
  ret double %a
}

; CHECK-LABEL: fmul64:
; CHECK: (setlocal @2 (fmul @1 @0))
define double @fmul64(double %x, double %y) {
  %a = fmul double %x, %y
  ret double %a
}

; CHECK-LABEL: fdiv64:
; CHECK: (setlocal @2 (fdiv @1 @0))
define double @fdiv64(double %x, double %y) {
  %a = fdiv double %x, %y
  ret double %a
}

; CHECK-LABEL: fabs64:
; CHECK: (setlocal @1 (fabs @0))
define double @fabs64(double %x) {
  %a = call double @llvm.fabs.f64(double %x)
  ret double %a
}

; CHECK-LABEL: fneg64:
; CHECK: (setlocal @1 (fneg @0))
define double @fneg64(double %x) {
  %a = fsub double -0., %x
  ret double %a
}

; CHECK-LABEL: copysign64:
; CHECK: (setlocal @2 (copysign @1 @0))
define double @copysign64(double %x, double %y) {
  %a = call double @llvm.copysign.f64(double %x, double %y)
  ret double %a
}

; CHECK-LABEL: sqrt64:
; CHECK: (setlocal @1 (sqrt @0))
define double @sqrt64(double %x) {
  %a = call double @llvm.sqrt.f64(double %x)
  ret double %a
}
