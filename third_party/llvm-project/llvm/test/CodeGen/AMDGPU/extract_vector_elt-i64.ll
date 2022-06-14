; RUN: llc -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; How the replacement of i64 stores with v2i32 stores resulted in
; breaking other users of the bitcast if they already existed

; GCN-LABEL: {{^}}extract_vector_elt_select_error:
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @extract_vector_elt_select_error(i32 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %val) #0 {
  %vec = bitcast i64 %val to <2 x i32>
  %elt0 = extractelement <2 x i32> %vec, i32 0
  %elt1 = extractelement <2 x i32> %vec, i32 1

  store volatile i32 %elt0, i32 addrspace(1)* %out
  store volatile i32 %elt1, i32 addrspace(1)* %out
  store volatile i64 %val, i64 addrspace(1)* %in
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i64:
define amdgpu_kernel void @extract_vector_elt_v2i64(i64 addrspace(1)* %out, <2 x i64> %foo) #0 {
  %p0 = extractelement <2 x i64> %foo, i32 0
  %p1 = extractelement <2 x i64> %foo, i32 1
  %out1 = getelementptr i64, i64 addrspace(1)* %out, i32 1
  store volatile i64 %p1, i64 addrspace(1)* %out
  store volatile i64 %p0, i64 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v2i64:
; GCN-NOT: buffer_load
; GCN-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; GCN-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @dyn_extract_vector_elt_v2i64(i64 addrspace(1)* %out, <2 x i64> %foo, i32 %elt) #0 {
  %dynelt = extractelement <2 x i64> %foo, i32 %elt
  store volatile i64 %dynelt, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v2i64_2:
; GCN:     buffer_load_dwordx4
; GCN-NOT: buffer_load
; GCN-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; GCN-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @dyn_extract_vector_elt_v2i64_2(i64 addrspace(1)* %out, <2 x i64> addrspace(1)* %foo, i32 %elt, <2 x i64> %arst) #0 {
  %load = load volatile <2 x i64>, <2 x i64> addrspace(1)* %foo
  %or = or <2 x i64> %load, %arst
  %dynelt = extractelement <2 x i64> %or, i32 %elt
  store volatile i64 %dynelt, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v3i64:
; GCN-NOT: buffer_load
; GCN-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; GCN-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; GCN-DAG: s_cmp_eq_u32 [[IDX]], 2
; GCN-DAG: s_cselect_b64 [[C2:[^,]+]], -1, 0
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @dyn_extract_vector_elt_v3i64(i64 addrspace(1)* %out, <3 x i64> %foo, i32 %elt) #0 {
  %dynelt = extractelement <3 x i64> %foo, i32 %elt
  store volatile i64 %dynelt, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v4i64:
; GCN-NOT: buffer_load
; GCN-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; GCN-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; GCN-DAG: s_cmp_eq_u32 [[IDX]], 2
; GCN-DAG: s_cselect_b64 [[C2:[^,]+]], -1, 0
; GCN-DAG: s_cmp_eq_u32 [[IDX]], 3
; GCN-DAG: s_cselect_b64 [[C3:[^,]+]], -1, 0
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C3]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C3]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @dyn_extract_vector_elt_v4i64(i64 addrspace(1)* %out, <4 x i64> %foo, i32 %elt) #0 {
  %dynelt = extractelement <4 x i64> %foo, i32 %elt
  store volatile i64 %dynelt, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
