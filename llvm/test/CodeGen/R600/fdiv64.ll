; RUN: llc < %s -march=r600 -mcpu=tahiti | FileCheck %s

; CHECK: @fdiv_f64
; CHECK: V_RCP_F64_e32 {{VGPR[0-9]+_VGPR[0-9]+}}
; CHECK: V_MUL_F64 {{VGPR[0-9]+_VGPR[0-9]+, VGPR[0-9]+_VGPR[0-9]+, VGPR[0-9]+_VGPR[0-9]+}}

define void @fdiv_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                      double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fdiv double %r0, %r1
   store double %r2, double addrspace(1)* %out
   ret void
}
