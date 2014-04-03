; RUN: llc < %s -march=r600 -mcpu=tahiti -verify-machineinstrs | FileCheck %s

; CHECK: @fconst_f64
; CHECK-DAG: S_MOV_B32 {{s[0-9]+}}, 1075052544
; CHECK-DAG: S_MOV_B32 {{s[0-9]+}}, 0

define void @fconst_f64(double addrspace(1)* %out, double addrspace(1)* %in) {
   %r1 = load double addrspace(1)* %in
   %r2 = fadd double %r1, 5.000000e+00
   store double %r2, double addrspace(1)* %out
   ret void
}
