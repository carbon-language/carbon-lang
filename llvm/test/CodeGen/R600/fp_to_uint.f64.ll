; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: @fp_to_uint_i32_f64
; SI: V_CVT_U32_F64_e32
define void @fp_to_uint_i32_f64(i32 addrspace(1)* %out, double %in) {
  %cast = fptoui double %in to i32
  store i32 %cast, i32 addrspace(1)* %out, align 4
  ret void
}
