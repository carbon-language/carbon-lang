; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}select_undef_lhs:
; GCN: s_waitcnt
; GCN-NOT: v_cmp
; GCN-NOT: v_cndmask
; GCN-NEXT: s_setpc_b64
define float @select_undef_lhs(float %val, i1 %cond) {
  %undef = call float @llvm.amdgcn.rcp.f32(float undef)
  %sel = select i1 %cond, float %undef, float %val
  ret float %sel
}

; GCN-LABEL: {{^}}select_undef_rhs:
; GCN: s_waitcnt
; GCN-NOT: v_cmp
; GCN-NOT: v_cndmask
; GCN-NEXT: s_setpc_b64
define float @select_undef_rhs(float %val, i1 %cond) {
  %undef = call float @llvm.amdgcn.rcp.f32(float undef)
  %sel = select i1 %cond, float %val, float %undef
  ret float %sel
}

declare float @llvm.amdgcn.rcp.f32(float)
