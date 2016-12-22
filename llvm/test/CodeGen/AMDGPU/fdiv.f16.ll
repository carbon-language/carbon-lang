; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; Make sure fdiv is promoted to f32.

; GCN-LABEL: {{^}}fdiv_f16
; SI:     v_cvt_f32_f16
; SI:     v_cvt_f32_f16
; SI:     v_div_scale_f32
; SI-DAG: v_div_scale_f32
; SI-DAG: v_rcp_f32
; SI:     v_fma_f32
; SI:     v_fma_f32
; SI:     v_mul_f32
; SI:     v_fma_f32
; SI:     v_fma_f32
; SI:     v_fma_f32
; SI:     v_div_fmas_f32
; SI:     v_div_fixup_f32
; SI:     v_cvt_f16_f32

; VI: buffer_load_ushort [[LHS:v[0-9]+]]
; VI: buffer_load_ushort [[RHS:v[0-9]+]]

; VI-DAG: v_cvt_f32_f16_e32 [[CVT_LHS:v[0-9]+]], [[LHS]]
; VI-DAG: v_cvt_f32_f16_e32 [[CVT_RHS:v[0-9]+]], [[RHS]]

; VI-DAG: v_rcp_f32_e32 [[RCP_RHS:v[0-9]+]], [[CVT_RHS]]
; VI: v_mul_f32_e32 [[MUL:v[0-9]+]], [[RCP_RHS]], [[CVT_LHS]]
; VI: v_cvt_f16_f32_e32 [[CVT_BACK:v[0-9]+]], [[MUL]]
; VI: v_div_fixup_f16 [[RESULT:v[0-9]+]], [[CVT_BACK]], [[RHS]], [[LHS]]
; VI: buffer_store_short [[RESULT]]
define void @fdiv_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load volatile half, half addrspace(1)* %a
  %b.val = load volatile half, half addrspace(1)* %b
  %r.val = fdiv half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}
