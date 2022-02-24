; RUN: llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX90A %s
; RUN: not --crash llc -global-isel < %s -march=amdgcn -mcpu=gfx908 -verify-machineinstrs 2>&1 | FileCheck %s -check-prefix=GFX908

declare float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float, <4 x i32>, i32, i32, i32, i32 immarg)
declare <2 x half> @llvm.amdgcn.struct.buffer.atomic.fadd.v2f16(<2 x half>, <4 x i32>, i32, i32, i32, i32 immarg)

; GFX908: error: {{.*}} return versions of fp atomics not supported

; GFX90A-LABEL: {{^}}buffer_atomic_add_f32_rtn:
; GFX90A: buffer_atomic_add_f32 v{{[0-9]+}}, v[{{[0-9:]+}}], s[{{[0-9:]+}}], s{{[0-9]+}} idxen offen glc
define amdgpu_kernel void @buffer_atomic_add_f32_rtn(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset) {
main_body:
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  store float %ret, float* undef
  ret void
}

; GFX90A-LABEL: {{^}}buffer_atomic_add_v2f16_rtn:
; GFX90A: buffer_atomic_pk_add_f16 v{{[0-9]+}}, v[{{[0-9:]+}}], s[{{[0-9:]+}}], s{{[0-9]+}} idxen offen glc
define amdgpu_kernel void @buffer_atomic_add_v2f16_rtn(<2 x half> %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset) {
main_body:
  %ret = call <2 x half> @llvm.amdgcn.struct.buffer.atomic.fadd.v2f16(<2 x half> %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  store <2 x half> %ret, <2 x half>* undef
  ret void
}
