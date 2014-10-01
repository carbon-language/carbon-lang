; RUN: llc < %s -march=r600 -mcpu=tahiti -verify-machineinstrs | FileCheck %s

; CHECK: {{^}}flt_f64:
; CHECK: V_CMP_LT_F64_e64 {{s[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+]}}

define void @flt_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fcmp ult double %r0, %r1
   %r3 = select i1 %r2, double %r0, double %r1
   store double %r3, double addrspace(1)* %out
   ret void
}

; CHECK: {{^}}fle_f64:
; CHECK: V_CMP_LE_F64_e64 {{s[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+]}}

define void @fle_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fcmp ule double %r0, %r1
   %r3 = select i1 %r2, double %r0, double %r1
   store double %r3, double addrspace(1)* %out
   ret void
}

; CHECK: {{^}}fgt_f64:
; CHECK: V_CMP_GT_F64_e64 {{s[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+]}}

define void @fgt_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fcmp ugt double %r0, %r1
   %r3 = select i1 %r2, double %r0, double %r1
   store double %r3, double addrspace(1)* %out
   ret void
}

; CHECK: {{^}}fge_f64:
; CHECK: V_CMP_GE_F64_e64 {{s[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+]}}

define void @fge_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fcmp uge double %r0, %r1
   %r3 = select i1 %r2, double %r0, double %r1
   store double %r3, double addrspace(1)* %out
   ret void
}

; CHECK: {{^}}fne_f64:
; CHECK: V_CMP_NEQ_F64_e32 vcc, {{v[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+]}}

define void @fne_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fcmp une double %r0, %r1
   %r3 = select i1 %r2, double %r0, double %r1
   store double %r3, double addrspace(1)* %out
   ret void
}

; CHECK: {{^}}feq_f64:
; CHECK: V_CMP_EQ_F64_e64 {{s[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+], v[[0-9]+:[0-9]+]}}

define void @feq_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                     double addrspace(1)* %in2) {
   %r0 = load double addrspace(1)* %in1
   %r1 = load double addrspace(1)* %in2
   %r2 = fcmp ueq double %r0, %r1
   %r3 = select i1 %r2, double %r0, double %r1
   store double %r3, double addrspace(1)* %out
   ret void
}
