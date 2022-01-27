; RUN: llc -global-isel -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V2,HSA,ALL %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=hawaii -mattr=+flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=CO-V2,OS-MESA3D,ALL %s
; RUN: llc -global-isel -mtriple=amdgcn-mesa-unknown -mcpu=hawaii -mattr=+flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=OS-UNKNOWN,ALL %s

; ALL-LABEL: {{^}}test:
; CO-V2: enable_sgpr_kernarg_segment_ptr = 1
; HSA: kernarg_segment_byte_size = 8
; HSA: kernarg_segment_alignment = 4

; CO-V2: s_load_dword s{{[0-9]+}}, s[4:5], 0xa

; OS-UNKNOWN: s_load_dword s{{[0-9]+}}, s[0:1], 0xa
define amdgpu_kernel void @test(i32 addrspace(1)* %out) #1 {
  %kernarg.segment.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %header.ptr = bitcast i8 addrspace(4)* %kernarg.segment.ptr to i32 addrspace(4)*
  %gep = getelementptr i32, i32 addrspace(4)* %header.ptr, i64 10
  %value = load i32, i32 addrspace(4)* %gep
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit:
; HSA: kernarg_segment_byte_size = 8
; OS-MESA3D: kernarg_segment_byte_size = 24
; CO-V2: kernarg_segment_alignment = 4

; 10 + 9 (36 prepended implicit bytes) + 2(out pointer) = 21 = 0x15

; OS-UNKNOWN: s_load_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, 0x15
define amdgpu_kernel void @test_implicit(i32 addrspace(1)* %out) #1 {
  %implicitarg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %header.ptr = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %gep = getelementptr i32, i32 addrspace(4)* %header.ptr, i64 10
  %value = load i32, i32 addrspace(4)* %gep
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_implicit_alignment:
; HSA: kernarg_segment_byte_size = 12
; OS-MESA3D: kernarg_segment_byte_size = 28
; CO-V2: kernarg_segment_alignment = 4


; OS-UNKNOWN: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xc
; HSA: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x4
; OS-MESA3D: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x3
; ALL: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[VAL]]
; ALL: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[V_VAL]]
define amdgpu_kernel void @test_implicit_alignment(i32 addrspace(1)* %out, <2 x i8> %in) #1 {
  %implicitarg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %arg.ptr = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %arg.ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment
; HSA: kernarg_segment_byte_size = 64
; OS-MESA3D: kernarg_segment_byte_size = 28
; CO-V2: kernarg_segment_alignment = 4


; OS-UNKNOWN: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xc
; HSA: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x4
; OS-MESA3D: s_load_dword [[VAL:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x3
; ALL: v_mov_b32_e32 [[V_VAL:v[0-9]+]], [[VAL]]
; ALL: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[V_VAL]]
define amdgpu_kernel void @opencl_test_implicit_alignment(i32 addrspace(1)* %out, <2 x i8> %in) #2 {
  %implicitarg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %arg.ptr = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %val = load i32, i32 addrspace(4)* %arg.ptr
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_no_kernargs:
; CO-V2: enable_sgpr_kernarg_segment_ptr = 0
; CO-V2: kernarg_segment_byte_size = 0
; CO-V2: kernarg_segment_alignment = 4

; HSA: s_mov_b64 [[OFFSET_NULL:s\[[0-9]+:[0-9]+\]]], 40{{$}}
; HSA: s_load_dword s{{[0-9]+}}, [[OFFSET_NULL]]
define amdgpu_kernel void @test_no_kernargs() #1 {
  %kernarg.segment.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %header.ptr = bitcast i8 addrspace(4)* %kernarg.segment.ptr to i32 addrspace(4)*
  %gep = getelementptr i32, i32 addrspace(4)* %header.ptr, i64 10
  %value = load i32, i32 addrspace(4)* %gep
  store volatile i32 %value, i32 addrspace(1)* undef
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment_no_explicit_kernargs:
; HSA: kernarg_segment_byte_size = 48
; OS-MESA3D: kernarg_segment_byte_size = 16
; CO-V2: kernarg_segment_alignment = 4
define amdgpu_kernel void @opencl_test_implicit_alignment_no_explicit_kernargs() #2 {
  %implicitarg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %arg.ptr = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %val = load volatile i32, i32 addrspace(4)* %arg.ptr
  store volatile i32 %val, i32 addrspace(1)* null
  ret void
}

; ALL-LABEL: {{^}}opencl_test_implicit_alignment_no_explicit_kernargs_round_up:
; HSA: kernarg_segment_byte_size = 40
; OS-MESA3D: kernarg_segment_byte_size = 16
; CO-V2: kernarg_segment_alignment = 4
define amdgpu_kernel void @opencl_test_implicit_alignment_no_explicit_kernargs_round_up() #3 {
  %implicitarg.ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr()
  %arg.ptr = bitcast i8 addrspace(4)* %implicitarg.ptr to i32 addrspace(4)*
  %val = load volatile i32, i32 addrspace(4)* %arg.ptr
  store volatile i32 %val, i32 addrspace(1)* null
  ret void
}

; ALL-LABEL: {{^}}func_kernarg_segment_ptr:
; ALL: v_mov_b32_e32 v0, 0{{$}}
; ALL: v_mov_b32_e32 v1, 0{{$}}
define i8 addrspace(4)* @func_kernarg_segment_ptr() {
  %ptr = call i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  ret i8 addrspace(4)* %ptr
}

declare i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #0
declare i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "amdgpu-implicitarg-num-bytes"="0" }
attributes #2 = { nounwind "amdgpu-implicitarg-num-bytes"="48" }
attributes #3 = { nounwind "amdgpu-implicitarg-num-bytes"="38" }
