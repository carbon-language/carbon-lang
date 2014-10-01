; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=SI %s

; Make sure there isn't an extra space between the instruction name and first operands.

; SI-LABEL: {{^}}add_f32:
; SI-DAG: S_LOAD_DWORD [[SREGA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: S_LOAD_DWORD [[SREGB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: V_MOV_B32_e32 [[VREGB:v[0-9]+]], [[SREGB]]
; SI: V_ADD_F32_e32 [[RESULT:v[0-9]+]], [[SREGA]], [[VREGB]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
define void @add_f32(float addrspace(1)* %out, float %a, float %b) {
  %result = fadd float %a, %b
  store float %result, float addrspace(1)* %out
  ret void
}
