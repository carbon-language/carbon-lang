; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI  %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=r600 -mcpu=cypress < %s | FileCheck -enable-var-scope -check-prefix=EG %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

define amdgpu_kernel void @trunc_i64_to_i32_store(i32 addrspace(1)* %out, [8 x i32], i64 %in) {
; GCN-LABEL: {{^}}trunc_i64_to_i32_store:
; GCN: s_load_dword [[SLOAD:s[0-9]+]], s[0:1],
; GCN: v_mov_b32_e32 [[VLOAD:v[0-9]+]], [[SLOAD]]
; SI: buffer_store_dword [[VLOAD]]
; VI: flat_store_dword v[{{[0-9:]+}}], [[VLOAD]]

; EG-LABEL: {{^}}trunc_i64_to_i32_store:
; EG: MEM_RAT_CACHELESS STORE_RAW T0.X, T1.X, 1
; EG: LSHR
; EG-NEXT: 2(

  %result = trunc i64 %in to i32 store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}trunc_load_shl_i64:
; GCN-DAG: s_load_dwordx2
; GCN-DAG: s_load_dword [[SREG:s[0-9]+]],
; GCN: s_lshl_b32 [[SHL:s[0-9]+]], [[SREG]], 2
; GCN: v_mov_b32_e32 [[VSHL:v[0-9]+]], [[SHL]]
; SI: buffer_store_dword [[VSHL]]
; VI: flat_store_dword v[{{[0-9:]+}}], [[VSHL]]

define amdgpu_kernel void @trunc_load_shl_i64(i32 addrspace(1)* %out, [8 x i32], i64 %a) {
  %b = shl i64 %a, 2
  %result = trunc i64 %b to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_i64:
; SI: s_load_dwordx2 s{{\[}}[[LO_SREG:[0-9]+]]:{{[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd
; VI: s_load_dwordx2 s{{\[}}[[LO_SREG:[0-9]+]]:{{[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x34
; GCN: s_lshl_b64 s{{\[}}[[LO_SHL:[0-9]+]]:{{[0-9]+\]}}, s{{\[}}[[LO_SREG]]:{{[0-9]+\]}}, 2
; GCN: s_add_u32 s[[LO_SREG2:[0-9]+]], s[[LO_SHL]],
; GCN: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], s[[LO_SREG2]]
; GCN: s_addc_u32
; SI: buffer_store_dword v[[LO_VREG]],
; VI: flat_store_dword v[{{[0-9:]+}}], v[[LO_VREG]]
; GCN: v_mov_b32_e32
; GCN: v_mov_b32_e32
define amdgpu_kernel void @trunc_shl_i64(i64 addrspace(1)* %out2, i32 addrspace(1)* %out, i64 %a) {
  %aa = add i64 %a, 234 ; Prevent shrinking store.
  %b = shl i64 %aa, 2
  %result = trunc i64 %b to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  store i64 %b, i64 addrspace(1)* %out2, align 8 ; Prevent reducing ops to 32-bits
  ret void
}

; GCN-LABEL: {{^}}trunc_i32_to_i1:
; GCN: v_and_b32_e32 [[VREG:v[0-9]+]], 1, v{{[0-9]+}}
define amdgpu_kernel void @trunc_i32_to_i1(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) {
  %a = load i32, i32 addrspace(1)* %ptr, align 4
  %trunc = trunc i32 %a to i1
  %result = select i1 %trunc, i32 1, i32 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}trunc_i8_to_i1:
; GCN: v_and_b32_e32 [[VREG:v[0-9]+]], 1, v{{[0-9]+}}
define amdgpu_kernel void @trunc_i8_to_i1(i8 addrspace(1)* %out, i8 addrspace(1)* %ptr) {
  %a = load i8, i8 addrspace(1)* %ptr, align 4
  %trunc = trunc i8 %a to i1
  %result = select i1 %trunc, i8 1, i8 0
  store i8 %result, i8 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sgpr_trunc_i16_to_i1:
; GCN: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 1
define amdgpu_kernel void @sgpr_trunc_i16_to_i1(i16 addrspace(1)* %out, i16 %a) {
  %trunc = trunc i16 %a to i1
  %result = select i1 %trunc, i16 1, i16 0
  store i16 %result, i16 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sgpr_trunc_i32_to_i1:
; GCN: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, 1
define amdgpu_kernel void @sgpr_trunc_i32_to_i1(i32 addrspace(1)* %out, i32 %a) {
  %trunc = trunc i32 %a to i1
  %result = select i1 %trunc, i32 1, i32 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}s_trunc_i64_to_i1:
; SI: s_load_dwordx2 s{{\[}}[[SLO:[0-9]+]]:{{[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x13
; VI: s_load_dwordx2 s{{\[}}[[SLO:[0-9]+]]:{{[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x4c
; GCN: s_and_b32 [[MASKED:s[0-9]+]], 1, s[[SLO]]
; GCN: v_cmp_eq_u32_e64 s{{\[}}[[VLO:[0-9]+]]:[[VHI:[0-9]+]]], [[MASKED]], 1{{$}}
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, -12, 63, s{{\[}}[[VLO]]:[[VHI]]]
; VI: s_cmp_lg_u64 s{{\[}}[[VLO]]:[[VHI]]], 0
; VI: s_cselect_b32 {{s[0-9]+}}, 63, -12
define amdgpu_kernel void @s_trunc_i64_to_i1(i32 addrspace(1)* %out, [8 x i32], i64 %x) {
  %trunc = trunc i64 %x to i1
  %sel = select i1 %trunc, i32 63, i32 -12
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_trunc_i64_to_i1:
; SI: buffer_load_dwordx2 v{{\[}}[[VLO:[0-9]+]]:{{[0-9]+\]}}
; VI: flat_load_dwordx2 v{{\[}}[[VLO:[0-9]+]]:{{[0-9]+\]}}
; GCN: v_and_b32_e32 [[MASKED:v[0-9]+]], 1, v[[VLO]]
; GCN: v_cmp_eq_u32_e32 vcc, 1, [[MASKED]]
; GCN: v_cndmask_b32_e64 {{v[0-9]+}}, -12, 63, vcc
define amdgpu_kernel void @v_trunc_i64_to_i1(i32 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %x = load i64, i64 addrspace(1)* %gep

  %trunc = trunc i64 %x to i1
  %sel = select i1 %trunc, i32 63, i32 -12
  store i32 %sel, i32 addrspace(1)* %out.gep
  ret void
}
