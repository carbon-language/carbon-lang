; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -strict-whitespace -check-prefix=SI %s

; Make sure there isn't an extra space between the instruction name and first operands.

; SI-LABEL: {{^}}add_f32:
; SI-DAG: s_load_dword [[SREGA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SREGB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VREGB:v[0-9]+]], [[SREGB]]
; SI: v_add_f32_e32 [[RESULT:v[0-9]+]], [[SREGA]], [[VREGB]]
; SI: buffer_store_dword [[RESULT]],
define void @add_f32(float addrspace(1)* %out, float %a, float %b) {
  %result = fadd float %a, %b
  store float %result, float addrspace(1)* %out
  ret void
}
