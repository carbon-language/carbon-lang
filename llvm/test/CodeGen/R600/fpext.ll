; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=CHECK

; CHECK: {{^}}fpext:
; CHECK: v_cvt_f64_f32_e32
define void @fpext(double addrspace(1)* %out, float %in) {
  %result = fpext float %in to double
  store double %result, double addrspace(1)* %out
  ret void
}
