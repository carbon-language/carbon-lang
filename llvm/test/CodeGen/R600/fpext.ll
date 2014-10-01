; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=CHECK

; CHECK: {{^}}fpext:
; CHECK: V_CVT_F64_F32_e32
define void @fpext(double addrspace(1)* %out, float %in) {
  %result = fpext float %in to double
  store double %result, double addrspace(1)* %out
  ret void
}
