; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=CHECK

; CHECK: {{^}}fptrunc:
; CHECK: v_cvt_f32_f64_e32
define void @fptrunc(float addrspace(1)* %out, double %in) {
  %result = fptrunc double %in to float
  store float %result, float addrspace(1)* %out
  ret void
}
