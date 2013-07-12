; RUN: llc < %s -march=r600 -mcpu=tahiti | FileCheck %s

; CHECK: @fconst_f64
; CHECK: V_MOV_B32_e32 {{VGPR[0-9]+}}, 0.000000e+00
; CHECK-NEXT: V_MOV_B32_e32 {{VGPR[0-9]+}}, 2.312500e+00

define void @fconst_f64(double addrspace(1)* %out, double addrspace(1)* %in) {
   %r1 = load double addrspace(1)* %in
   %r2 = fadd double %r1, 5.000000e+00
   store double %r2, double addrspace(1)* %out
   ret void
}
