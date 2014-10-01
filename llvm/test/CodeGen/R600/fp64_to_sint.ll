; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}fp_to_sint_f64_i32:
; SI: V_CVT_I32_F64_e32
define void @fp_to_sint_f64_i32(i32 addrspace(1)* %out, double %in) {
  %result = fptosi double %in to i32
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_sint_v2f64_v2i32:
; SI: V_CVT_I32_F64_e32
; SI: V_CVT_I32_F64_e32
define void @fp_to_sint_v2f64_v2i32(<2 x i32> addrspace(1)* %out, <2 x double> %in) {
  %result = fptosi <2 x double> %in to <2 x i32>
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_sint_v4f64_v4i32:
; SI: V_CVT_I32_F64_e32
; SI: V_CVT_I32_F64_e32
; SI: V_CVT_I32_F64_e32
; SI: V_CVT_I32_F64_e32
define void @fp_to_sint_v4f64_v4i32(<4 x i32> addrspace(1)* %out, <4 x double> %in) {
  %result = fptosi <4 x double> %in to <4 x i32>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}
