; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -enable-var-scope -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}s_fneg_f32:
; R600: -PV

; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_xor_b32 [[NEG_VAL:s[0-9]+]], [[VAL]], 0x80000000
; GCN: v_mov_b32_e32 v{{[0-9]+}}, [[NEG_VAL]]
define amdgpu_kernel void @s_fneg_f32(float addrspace(1)* %out, float %in) {
  %fneg = fsub float -0.000000e+00, %in
  store float %fneg, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_fneg_v2f32:
; R600: -PV
; R600: -PV

; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
define amdgpu_kernel void @s_fneg_v2f32(<2 x float> addrspace(1)* nocapture %out, <2 x float> %in) {
  %fneg = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %in
  store <2 x float> %fneg, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_fneg_v4f32:
; R600: -PV
; R600: -T
; R600: -PV
; R600: -PV

; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
define amdgpu_kernel void @s_fneg_v4f32(<4 x float> addrspace(1)* nocapture %out, <4 x float> %in) {
  %fneg = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %in
  store <4 x float> %fneg, <4 x float> addrspace(1)* %out
  ret void
}

; DAGCombiner will transform:
; (fneg (f32 bitcast (i32 a))) => (f32 bitcast (xor (i32 a), 0x80000000))
; unless the target returns true for isNegFree()

; FUNC-LABEL: {{^}}fsub0_f32:

; GCN: v_sub_f32_e64 v{{[0-9]}}, 0, s{{[0-9]+$}}

; R600-NOT: XOR
; R600: -KC0[2].Z
define amdgpu_kernel void @fsub0_f32(float addrspace(1)* %out, i32 %in) {
  %bc = bitcast i32 %in to float
  %fsub = fsub float 0.0, %bc
  store float %fsub, float addrspace(1)* %out
  ret void
}
; FUNC-LABEL: {{^}}fneg_free_f32:
; SI: s_load_dword [[NEG_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[NEG_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c

; GCN: s_xor_b32 [[RES:s[0-9]+]], [[NEG_VALUE]], 0x80000000
; GCN: v_mov_b32_e32 [[V_RES:v[0-9]+]], [[RES]]
; GCN: buffer_store_dword [[V_RES]]

; R600-NOT: XOR
; R600: -PV.W
define amdgpu_kernel void @fneg_free_f32(float addrspace(1)* %out, i32 %in) {
  %bc = bitcast i32 %in to float
  %fsub = fsub float -0.0, %bc
  store float %fsub, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fneg_fold_f32:
; SI: s_load_dword [[NEG_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[NEG_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; GCN-NOT: xor
; GCN: v_mul_f32_e64 v{{[0-9]+}}, -[[NEG_VALUE]], [[NEG_VALUE]]
define amdgpu_kernel void @fneg_fold_f32(float addrspace(1)* %out, float %in) {
  %fsub = fsub float -0.0, %in
  %fmul = fmul float %fsub, %in
  store float %fmul, float addrspace(1)* %out
  ret void
}

; Make sure we turn some integer operations back into fabs
; FUNC-LABEL: {{^}}bitpreserve_fneg_f32:
; GCN: v_mul_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, -4.0
define amdgpu_kernel void @bitpreserve_fneg_f32(float addrspace(1)* %out, float %in) {
  %in.bc = bitcast float %in to i32
  %int.abs = xor i32 %in.bc, 2147483648
  %bc = bitcast i32 %int.abs to float
  %fadd = fmul float %bc, 4.0
  store float %fadd, float addrspace(1)* %out
  ret void
}
