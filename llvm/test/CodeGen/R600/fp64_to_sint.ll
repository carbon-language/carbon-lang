; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=CHECK

; CHECK: @fp64_to_sint
; CHECK: V_CVT_I32_F64_e32
define void @fp64_to_sint(i32 addrspace(1)* %out, double %in) {
  %result = fptosi double %in to i32
  store i32 %result, i32 addrspace(1)* %out
  ret void
}
