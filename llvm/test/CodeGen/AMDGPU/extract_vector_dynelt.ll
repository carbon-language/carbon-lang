; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s

; GCN-LABEL: {{^}}bit4_extelt:
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN-DAG: buffer_store_byte [[ZERO]],
; GCN-DAG: buffer_store_byte [[ONE]],
; GCN-DAG: buffer_store_byte [[ZERO]],
; GCN-DAG: buffer_store_byte [[ONE]],
; GCN:     buffer_load_ubyte [[LOAD:v[0-9]+]],
; GCN:     v_and_b32_e32 [[RES:v[0-9]+]], 1, [[LOAD]]
; GCN:     flat_store_dword v[{{[0-9:]+}}], [[RES]]
define amdgpu_kernel void @bit4_extelt(i32 addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <4 x i1> <i1 0, i1 1, i1 0, i1 1>, i32 %sel
  %zext = zext i1 %ext to i32
  store i32 %zext, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}bit128_extelt:
define amdgpu_kernel void @bit128_extelt(i32 addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <128 x i1> <i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0>, i32 %sel
  %zext = zext i1 %ext to i32
  store i32 %zext, i32 addrspace(1)* %out
  ret void
}
