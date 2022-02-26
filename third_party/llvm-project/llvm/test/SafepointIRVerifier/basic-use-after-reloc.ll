; RUN: opt -safepoint-ir-verifier-print-only -verify-safepoint-ir -S %s 2>&1 | FileCheck %s

; This test checks that if a value is used immediately after a
; safepoint without using the relocated value that the verifier
; catches this.

%jObject = type { [8 x i8] }

; Function Attrs: nounwind
define %jObject addrspace(1)* @test(%jObject addrspace(1)* %arg) gc "statepoint-example" {
bci_0:
  %safepoint_token3 = tail call token (i64, i32, double (double)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_f64f64f(i64 0, i32 0, double (double)* elementtype(double (double)) undef, i32 1, i32 0, double undef, i32 0, i32 0) ["gc-live"(%jObject addrspace(1)* %arg)]
  %arg2.relocated4 = call coldcc %jObject addrspace(1)* @llvm.experimental.gc.relocate.p1jObject(token %safepoint_token3, i32 0, i32 0)
  ret %jObject addrspace(1)* %arg
; CHECK: Illegal use of unrelocated value found!
; CHECK-NEXT: Def: %jObject addrspace(1)* %arg
; CHECK-NEXT: Use:   ret %jObject addrspace(1)* %arg
}

; Function Attrs: nounwind
declare %jObject addrspace(1)* @llvm.experimental.gc.relocate.p1jObject(token, i32, i32) #3

declare token @llvm.experimental.gc.statepoint.p0f_f64f64f(i64, i32, double (double)*, i32, i32, ...)
