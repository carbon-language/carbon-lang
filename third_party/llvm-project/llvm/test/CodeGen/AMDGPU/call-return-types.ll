; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX7 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX89 %s

declare void @external_void_func_void() #0

declare i1 @external_i1_func_void() #0
declare zeroext i1 @external_i1_zeroext_func_void() #0
declare signext i1 @external_i1_signext_func_void() #0

declare i8 @external_i8_func_void() #0
declare zeroext i8 @external_i8_zeroext_func_void() #0
declare signext i8 @external_i8_signext_func_void() #0

declare i16 @external_i16_func_void() #0
declare <2 x i16> @external_v2i16_func_void() #0
declare <4 x i16> @external_v4i16_func_void() #0
declare zeroext i16 @external_i16_zeroext_func_void() #0
declare signext i16 @external_i16_signext_func_void() #0

declare i32 @external_i32_func_void() #0
declare i64 @external_i64_func_void() #0
declare half @external_f16_func_void() #0
declare float @external_f32_func_void() #0
declare double @external_f64_func_void() #0

declare <2 x half> @external_v2f16_func_void() #0
declare <4 x half> @external_v4f16_func_void() #0
declare <3 x float> @external_v3f32_func_void() #0
declare <5 x float> @external_v5f32_func_void() #0
declare <2 x double> @external_v2f64_func_void() #0

declare <2 x i24> @external_v2i24_func_void() #0

declare <2 x i32> @external_v2i32_func_void() #0
declare <3 x i32> @external_v3i32_func_void() #0
declare <4 x i32> @external_v4i32_func_void() #0
declare <5 x i32> @external_v5i32_func_void() #0
declare <8 x i32> @external_v8i32_func_void() #0
declare <16 x i32> @external_v16i32_func_void() #0
declare <32 x i32> @external_v32i32_func_void() #0
declare { <32 x i32>, i32 } @external_v32i32_i32_func_void() #0

declare { i32, i64 } @external_i32_i64_func_void() #0

; GCN-LABEL: {{^}}test_call_external_void_func_void:
define amdgpu_kernel void @test_call_external_void_func_void() #0 {
  call void @external_void_func_void()
  ret void
}

; GCN-LABEL: {{^}}test_call_external_void_func_void_x2:
define amdgpu_kernel void @test_call_external_void_func_void_x2() #0 {
  call void @external_void_func_void()
  call void @external_void_func_void()
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i1_func_void:
define amdgpu_kernel void @test_call_external_i1_func_void() #0 {
  %val = call i1 @external_i1_func_void()
  store volatile i1 %val, i1 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i1_zeroext_func_void:
define amdgpu_kernel void @test_call_external_i1_zeroext_func_void() #0 {
  %val = call i1 @external_i1_zeroext_func_void()
  %val.ext = zext i1 %val to i32
  store volatile i32 %val.ext, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i1_signext_func_void:
define amdgpu_kernel void @test_call_external_i1_signext_func_void() #0 {
  %val = call i1 @external_i1_signext_func_void()
  %val.ext = zext i1 %val to i32
  store volatile i32 %val.ext, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i8_func_void:
define amdgpu_kernel void @test_call_external_i8_func_void() #0 {
  %val = call i8 @external_i8_func_void()
  store volatile i8 %val, i8 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i8_zeroext_func_void:
define amdgpu_kernel void @test_call_external_i8_zeroext_func_void() #0 {
  %val = call i8 @external_i8_zeroext_func_void()
  %val.ext = zext i8 %val to i32
  store volatile i32 %val.ext, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i8_signext_func_void:
define amdgpu_kernel void @test_call_external_i8_signext_func_void() #0 {
  %val = call i8 @external_i8_signext_func_void()
  %val.ext = zext i8 %val to i32
  store volatile i32 %val.ext, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i16_func_void:
define amdgpu_kernel void @test_call_external_i16_func_void() #0 {
  %val = call i16 @external_i16_func_void()
  store volatile i16 %val, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i16_zeroext_func_void:
define amdgpu_kernel void @test_call_external_i16_zeroext_func_void() #0 {
  %val = call i16 @external_i16_zeroext_func_void()
  %val.ext = zext i16 %val to i32
  store volatile i32 %val.ext, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i16_signext_func_void:
define amdgpu_kernel void @test_call_external_i16_signext_func_void() #0 {
  %val = call i16 @external_i16_signext_func_void()
  %val.ext = zext i16 %val to i32
  store volatile i32 %val.ext, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i32_func_void:
define amdgpu_kernel void @test_call_external_i32_func_void() #0 {
  %val = call i32 @external_i32_func_void()
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i64_func_void:
define amdgpu_kernel void @test_call_external_i64_func_void() #0 {
  %val = call i64 @external_i64_func_void()
  store volatile i64 %val, i64 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_f16_func_void:
define amdgpu_kernel void @test_call_external_f16_func_void() #0 {
  %val = call half @external_f16_func_void()
  store volatile half %val, half addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_f32_func_void:
define amdgpu_kernel void @test_call_external_f32_func_void() #0 {
  %val = call float @external_f32_func_void()
  store volatile float %val, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_f64_func_void:
define amdgpu_kernel void @test_call_external_f64_func_void() #0 {
  %val = call double @external_f64_func_void()
  store volatile double %val, double addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v2f64_func_void:
define amdgpu_kernel void @test_call_external_v2f64_func_void() #0 {
  %val = call <2 x double> @external_v2f64_func_void()
  store volatile <2 x double> %val, <2 x double> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v2i32_func_void:
define amdgpu_kernel void @test_call_external_v2i32_func_void() #0 {
  %val = call <2 x i32> @external_v2i32_func_void()
  store volatile <2 x i32> %val, <2 x i32> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v3i32_func_void:
; GCN: s_swappc
; GFX7-DAG: flat_store_dwordx3 {{.*}}, v[0:2]
; GFX89-DAG: buffer_store_dwordx3 v[0:2]
define amdgpu_kernel void @test_call_external_v3i32_func_void() #0 {
  %val = call <3 x i32> @external_v3i32_func_void()
  store volatile <3 x i32> %val, <3 x i32> addrspace(1)* undef, align 8
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v4i32_func_void:
define amdgpu_kernel void @test_call_external_v4i32_func_void() #0 {
  %val = call <4 x i32> @external_v4i32_func_void()
  store volatile <4 x i32> %val, <4 x i32> addrspace(1)* undef, align 8
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v5i32_func_void:
; GCN: s_swappc
; GFX7-DAG: flat_store_dwordx4 {{.*}}, v[0:3]
; GFX7-DAG: flat_store_dword {{.*}}, v4
; GFX89-DAG: buffer_store_dwordx4 v[0:3]
; GFX89-DAG: buffer_store_dword v4
define amdgpu_kernel void @test_call_external_v5i32_func_void() #0 {
  %val = call <5 x i32> @external_v5i32_func_void()
  store volatile <5 x i32> %val, <5 x i32> addrspace(1)* undef, align 8
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v8i32_func_void:
define amdgpu_kernel void @test_call_external_v8i32_func_void() #0 {
  %val = call <8 x i32> @external_v8i32_func_void()
  store volatile <8 x i32> %val, <8 x i32> addrspace(1)* undef, align 8
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v16i32_func_void:
define amdgpu_kernel void @test_call_external_v16i32_func_void() #0 {
  %val = call <16 x i32> @external_v16i32_func_void()
  store volatile <16 x i32> %val, <16 x i32> addrspace(1)* undef, align 8
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v32i32_func_void:
define amdgpu_kernel void @test_call_external_v32i32_func_void() #0 {
  %val = call <32 x i32> @external_v32i32_func_void()
  store volatile <32 x i32> %val, <32 x i32> addrspace(1)* undef, align 8
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v2i16_func_void:
define amdgpu_kernel void @test_call_external_v2i16_func_void() #0 {
  %val = call <2 x i16> @external_v2i16_func_void()
  store volatile <2 x i16> %val, <2 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v4i16_func_void:
define amdgpu_kernel void @test_call_external_v4i16_func_void() #0 {
  %val = call <4 x i16> @external_v4i16_func_void()
  store volatile <4 x i16> %val, <4 x i16> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v2f16_func_void:
define amdgpu_kernel void @test_call_external_v2f16_func_void() #0 {
  %val = call <2 x half> @external_v2f16_func_void()
  store volatile <2 x half> %val, <2 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v4f16_func_void:
define amdgpu_kernel void @test_call_external_v4f16_func_void() #0 {
  %val = call <4 x half> @external_v4f16_func_void()
  store volatile <4 x half> %val, <4 x half> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v2i24_func_void:
; GCN: s_swappc_b64
; GCN: v_add_{{i|u}}32_e32 v0, {{(vcc, )?}}v0, v1
define amdgpu_kernel void @test_call_external_v2i24_func_void() #0 {
  %val = call <2 x i24> @external_v2i24_func_void()
  %elt0 = extractelement <2 x i24> %val, i32 0
  %elt1 = extractelement <2 x i24> %val, i32 1
  %add = add i24 %elt0, %elt1
  store volatile i24 %add, i24 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v3f32_func_void:
; GCN: s_swappc
; GFX7-DAG: flat_store_dwordx3 {{.*}}, v[0:2]
; GFX89-DAG: buffer_store_dwordx3 v[0:2]
define amdgpu_kernel void @test_call_external_v3f32_func_void() #0 {
  %val = call <3 x float> @external_v3f32_func_void()
  store volatile <3 x float> %val, <3 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_v5f32_func_void:
; GCN: s_swappc
; GFX7-DAG: flat_store_dwordx4 {{.*}}, v[0:3]
; GFX7-DAG: flat_store_dword {{.*}}, v4
; GFX89-DAG: buffer_store_dwordx4 v[0:3]
; GFX89-DAG: buffer_store_dword v4
define amdgpu_kernel void @test_call_external_v5f32_func_void() #0 {
  %val = call <5 x float> @external_v5f32_func_void()
  store volatile <5 x float> %val, <5 x float> addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}test_call_external_i32_i64_func_void:
define amdgpu_kernel void @test_call_external_i32_i64_func_void() #0 {
  %val = call { i32, i64 } @external_i32_i64_func_void()
  %val.0 = extractvalue { i32, i64 } %val, 0
  %val.1 = extractvalue { i32, i64 } %val, 1
  store volatile i32 %val.0, i32 addrspace(1)* undef
  store volatile i64 %val.1, i64 addrspace(1)* undef
  ret void
}

; Requires writing results to stack
; GCN-LABEL: {{^}}test_call_external_v32i32_i32_func_void:
define amdgpu_kernel void @test_call_external_v32i32_i32_func_void() #0 {
  %val = call { <32 x i32>, i32 } @external_v32i32_i32_func_void()
  %val0 = extractvalue { <32 x i32>, i32 } %val, 0
  %val1 = extractvalue { <32 x i32>, i32 } %val, 1
  store volatile <32 x i32> %val0, <32 x i32> addrspace(1)* undef, align 8
  store volatile i32 %val1, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind noinline }
