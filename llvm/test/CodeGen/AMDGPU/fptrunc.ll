; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}fptrunc_f64_to_f32:
; SI: v_cvt_f32_f64_e32 {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}
define void @fptrunc_f64_to_f32(float addrspace(1)* %out, double %in) {
  %result = fptrunc double %in to float
  store float %result, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fptrunc_v2f64_to_v2f32:
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
define void @fptrunc_v2f64_to_v2f32(<2 x float> addrspace(1)* %out, <2 x double> %in) {
  %result = fptrunc <2 x double> %in to <2 x float>
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fptrunc_v4f64_to_v4f32:
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
define void @fptrunc_v4f64_to_v4f32(<4 x float> addrspace(1)* %out, <4 x double> %in) {
  %result = fptrunc <4 x double> %in to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fptrunc_v8f64_to_v8f32:
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
; SI: v_cvt_f32_f64_e32
define void @fptrunc_v8f64_to_v8f32(<8 x float> addrspace(1)* %out, <8 x double> %in) {
  %result = fptrunc <8 x double> %in to <8 x float>
  store <8 x float> %result, <8 x float> addrspace(1)* %out
  ret void
}
