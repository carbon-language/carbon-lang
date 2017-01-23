; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}extract_vector_elt_v2i16:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_store_short
; GCN: buffer_store_short
define void @extract_vector_elt_v2i16(i16 addrspace(1)* %out, <2 x i16> %foo) #0 {
  %p0 = extractelement <2 x i16> %foo, i32 0
  %p1 = extractelement <2 x i16> %foo, i32 1
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 10
  store i16 %p1, i16 addrspace(1)* %out, align 2
  store i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i16_dynamic_sgpr:
; GCN: s_load_dword [[VEC:s[0-9]+]]
; GCN: s_load_dword [[IDX:s[0-9]+]]
; GCN: s_lshr_b32 s{{[0-9]+}}, [[IDX]], 16
; GCN: v_mov_b32_e32 [[VVEC:v[0-9]+]], [[VEC]]
define void @extract_vector_elt_v2i16_dynamic_sgpr(i16 addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i32 %idx) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %elt = extractelement <2 x i16> %vec, i32 %idx
  store i16 %elt, i16 addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i16_dynamic_vgpr:
; GCN: {{buffer|flat}}_load_dword [[IDX:v[0-9]+]]
; GCN: buffer_load_dword [[VEC:v[0-9]+]]
; GCN: v_lshrrev_b32_e32 [[ELT:v[0-9]+]], 16, [[VEC]]
define void @extract_vector_elt_v2i16_dynamic_vgpr(i16 addrspace(1)* %out, <2 x i16> addrspace(1)* %vec.ptr, i32 addrspace(1)* %idx.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %idx.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i16, i16 addrspace(1)* %out, i64 %tid.ext
  %idx = load volatile i32, i32 addrspace(1)* %gep
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vec.ptr
  %elt = extractelement <2 x i16> %vec, i32 %idx
  store i16 %elt, i16 addrspace(1)* %out.gep, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v3i16:
; GCN: buffer_load_ushort
; GCN: buffer_store_short
; GCN: buffer_store_short
define void @extract_vector_elt_v3i16(i16 addrspace(1)* %out, <3 x i16> %foo) #0 {
  %p0 = extractelement <3 x i16> %foo, i32 0
  %p1 = extractelement <3 x i16> %foo, i32 2
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p1, i16 addrspace(1)* %out, align 2
  store i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v4i16:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_store_short
; GCN: buffer_store_short
define void @extract_vector_elt_v4i16(i16 addrspace(1)* %out, <4 x i16> %foo) #0 {
  %p0 = extractelement <4 x i16> %foo, i32 0
  %p1 = extractelement <4 x i16> %foo, i32 2
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 10
  store i16 %p1, i16 addrspace(1)* %out, align 2
  store i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v3i16:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort

; GCN: buffer_store_short
; GCN: buffer_store_short
; GCN: buffer_store_short

; GCN: buffer_load_ushort
; GCN: buffer_store_short
define void @dynamic_extract_vector_elt_v3i16(i16 addrspace(1)* %out, <3 x i16> %foo, i32 %idx) #0 {
  %p0 = extractelement <3 x i16> %foo, i32 %idx
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p0, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v4i16:
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort

; GCN: buffer_store_short
; GCN: buffer_store_short
; GCN: buffer_store_short
; GCN: buffer_store_short

; GCN: buffer_load_ushort
; GCN: buffer_store_short
define void @dynamic_extract_vector_elt_v4i16(i16 addrspace(1)* %out, <4 x i16> %foo, i32 %idx) #0 {
  %p0 = extractelement <4 x i16> %foo, i32 %idx
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p0, i16 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
