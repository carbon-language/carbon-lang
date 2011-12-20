; RUN: llc < %s -mcpu=cortex-a8 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

; CHECK: f
; This function is forced to spill a double.
; Verify that the spill slot is properly aligned.
;
; The caller-saved r4 is used as a scratch register for stack realignment.
; CHECK: push {r4, r7, lr}
; CHECK: bic r4, r4, #7
; CHECK: mov sp, r4
define void @f(double* nocapture %p) nounwind ssp {
entry:
  %0 = load double* %p, align 4
  tail call void asm sideeffect "", "~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15}"() nounwind
  tail call void @g() nounwind
  store double %0, double* %p, align 4
  ret void
}

declare void @g()
