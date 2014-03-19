; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck %s

define void @use_gep_address_space([1024 x i32] addrspace(3)* %array) nounwind {
; CHECK-LABEL: @use_gep_address_space:
; CHECK: V_MOV_B32_e32 [[PTR:v[0-9]+]], s{{[0-9]+}}
; CHECK: DS_WRITE_B32 [[PTR]], v{{[0-9]+}}, 64
  %p = getelementptr [1024 x i32] addrspace(3)* %array, i16 0, i16 16
  store i32 99, i32 addrspace(3)* %p
  ret void
}

define void @use_gep_address_space_large_offset([1024 x i32] addrspace(3)* %array) nounwind {
; CHECK-LABEL: @use_gep_address_space_large_offset:
; CHECK: S_ADD_I32
; CHECK: DS_WRITE_B32
  %p = getelementptr [1024 x i32] addrspace(3)* %array, i16 0, i16 16384
  store i32 99, i32 addrspace(3)* %p
  ret void
}

define void @gep_as_vector_v4(<4 x [1024 x i32] addrspace(3)*> %array) nounwind {
; CHECK-LABEL: @gep_as_vector_v4:
; CHECK: S_ADD_I32
; CHECK: S_ADD_I32
; CHECK: S_ADD_I32
; CHECK: S_ADD_I32
  %p = getelementptr <4 x [1024 x i32] addrspace(3)*> %array, <4 x i16> zeroinitializer, <4 x i16> <i16 16, i16 16, i16 16, i16 16>
  %p0 = extractelement <4 x i32 addrspace(3)*> %p, i32 0
  %p1 = extractelement <4 x i32 addrspace(3)*> %p, i32 1
  %p2 = extractelement <4 x i32 addrspace(3)*> %p, i32 2
  %p3 = extractelement <4 x i32 addrspace(3)*> %p, i32 3
  store i32 99, i32 addrspace(3)* %p0
  store i32 99, i32 addrspace(3)* %p1
  store i32 99, i32 addrspace(3)* %p2
  store i32 99, i32 addrspace(3)* %p3
  ret void
}

define void @gep_as_vector_v2(<2 x [1024 x i32] addrspace(3)*> %array) nounwind {
; CHECK-LABEL: @gep_as_vector_v2:
; CHECK: S_ADD_I32
; CHECK: S_ADD_I32
  %p = getelementptr <2 x [1024 x i32] addrspace(3)*> %array, <2 x i16> zeroinitializer, <2 x i16> <i16 16, i16 16>
  %p0 = extractelement <2 x i32 addrspace(3)*> %p, i32 0
  %p1 = extractelement <2 x i32 addrspace(3)*> %p, i32 1
  store i32 99, i32 addrspace(3)* %p0
  store i32 99, i32 addrspace(3)* %p1
  ret void
}

