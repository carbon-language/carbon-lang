; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit floating-point operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare double @llvm.fabs.f64(double)
declare double @llvm.copysign.f64(double, double)
declare double @llvm.sqrt.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.trunc.f64(double)
declare double @llvm.nearbyint.f64(double)
declare double @llvm.rint.f64(double)

; CHECK-LABEL: (func $fadd64
; CHECK-NEXT: (param f64) (param f64) (result f64)
; CHECK-NEXT: (setlocal @0 (argument 1))
; CHECK-NEXT: (setlocal @1 (argument 0))
; CHECK-NEXT: (setlocal @2 (fadd @1 @0))
; CHECK-NEXT: (return @2)
define double @fadd64(double %x, double %y) {
  %a = fadd double %x, %y
  ret double %a
}

; CHECK-LABEL: (func $fsub64
; CHECK: (setlocal @2 (fsub @1 @0))
define double @fsub64(double %x, double %y) {
  %a = fsub double %x, %y
  ret double %a
}

; CHECK-LABEL: (func $fmul64
; CHECK: (setlocal @2 (fmul @1 @0))
define double @fmul64(double %x, double %y) {
  %a = fmul double %x, %y
  ret double %a
}

; CHECK-LABEL: (func $fdiv64
; CHECK: (setlocal @2 (fdiv @1 @0))
define double @fdiv64(double %x, double %y) {
  %a = fdiv double %x, %y
  ret double %a
}

; CHECK-LABEL: (func $fabs64
; CHECK: (setlocal @1 (fabs @0))
define double @fabs64(double %x) {
  %a = call double @llvm.fabs.f64(double %x)
  ret double %a
}

; CHECK-LABEL: (func $fneg64
; CHECK: (setlocal @1 (fneg @0))
define double @fneg64(double %x) {
  %a = fsub double -0., %x
  ret double %a
}

; CHECK-LABEL: (func $copysign64
; CHECK: (setlocal @2 (copysign @1 @0))
define double @copysign64(double %x, double %y) {
  %a = call double @llvm.copysign.f64(double %x, double %y)
  ret double %a
}

; CHECK-LABEL: (func $sqrt64
; CHECK: (setlocal @1 (sqrt @0))
define double @sqrt64(double %x) {
  %a = call double @llvm.sqrt.f64(double %x)
  ret double %a
}

; CHECK-LABEL: (func $ceil64
; CHECK: (setlocal @1 (ceil @0))
define double @ceil64(double %x) {
  %a = call double @llvm.ceil.f64(double %x)
  ret double %a
}

; CHECK-LABEL: (func $floor64
; CHECK: (setlocal @1 (floor @0))
define double @floor64(double %x) {
  %a = call double @llvm.floor.f64(double %x)
  ret double %a
}

; CHECK-LABEL: (func $trunc64
; CHECK: (setlocal @1 (trunc @0))
define double @trunc64(double %x) {
  %a = call double @llvm.trunc.f64(double %x)
  ret double %a
}

; CHECK-LABEL: (func $nearestint64
; CHECK: (setlocal @1 (nearestint @0))
define double @nearestint64(double %x) {
  %a = call double @llvm.nearbyint.f64(double %x)
  ret double %a
}

; CHECK-LABEL: (func $nearestint64_via_rint
; CHECK: (setlocal @1 (nearestint @0))
define double @nearestint64_via_rint(double %x) {
  %a = call double @llvm.rint.f64(double %x)
  ret double %a
}
