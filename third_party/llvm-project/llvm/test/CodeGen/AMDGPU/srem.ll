; RUN: llc -amdgpu-scalarize-global-loads=false --verify-machineinstrs -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=SI,GFX9 -check-prefix=FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=r600 -mcpu=redwood -verify-machineinstrs < %s

; FUNC-LABEL: {{^}}srem_i16_7:
; GFX9: s_movk_i32 {{s[0-9]+}}, 0x4925
; GFX9: v_mul_lo_u32
define amdgpu_kernel void @srem_i16_7(i16 addrspace(1)* %out, i16 addrspace(1)* %in) {
  %num = load i16, i16 addrspace(1) * %in
  %result = srem i16 %num, 7
  store i16 %result, i16 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %num = load i32, i32 addrspace(1) * %in
  %den = load i32, i32 addrspace(1) * %den_ptr
  %result = srem i32 %num, %den
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_i32_4(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %num = load i32, i32 addrspace(1) * %in
  %result = srem i32 %num, 4
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

; FIXME: uniform i16 srem should not use VALU instructions
; FUNC-LABEL: {{^}}srem_i32_7:
; SI: s_mov_b32 [[MAGIC:s[0-9]+]], 0x92492493
; SI: v_mul_hi_i32 {{v[0-9]+}}, {{v[0-9]+}}, [[MAGIC]]
; SI: v_mul_lo_u32
; SI: v_sub_{{[iu]}}32
; SI: s_endpgm
define amdgpu_kernel void @srem_i32_7(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %num = load i32, i32 addrspace(1) * %in
  %result = srem i32 %num, 7
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %den_ptr = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 1
  %num = load <2 x i32>, <2 x i32> addrspace(1) * %in
  %den = load <2 x i32>, <2 x i32> addrspace(1) * %den_ptr
  %result = srem <2 x i32> %num, %den
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_v2i32_4(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %num = load <2 x i32>, <2 x i32> addrspace(1) * %in
  %result = srem <2 x i32> %num, <i32 4, i32 4>
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %den_ptr = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %in, i32 1
  %num = load <4 x i32>, <4 x i32> addrspace(1) * %in
  %den = load <4 x i32>, <4 x i32> addrspace(1) * %den_ptr
  %result = srem <4 x i32> %num, %den
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_v4i32_4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %num = load <4 x i32>, <4 x i32> addrspace(1) * %in
  %result = srem <4 x i32> %num, <i32 4, i32 4, i32 4, i32 4>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %den_ptr = getelementptr i64, i64 addrspace(1)* %in, i64 1
  %num = load i64, i64 addrspace(1) * %in
  %den = load i64, i64 addrspace(1) * %den_ptr
  %result = srem i64 %num, %den
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_i64_4(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %num = load i64, i64 addrspace(1) * %in
  %result = srem i64 %num, 4
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %den_ptr = getelementptr <2 x i64>, <2 x i64> addrspace(1)* %in, i64 1
  %num = load <2 x i64>, <2 x i64> addrspace(1) * %in
  %den = load <2 x i64>, <2 x i64> addrspace(1) * %den_ptr
  %result = srem <2 x i64> %num, %den
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_v2i64_4(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %in) {
  %num = load <2 x i64>, <2 x i64> addrspace(1) * %in
  %result = srem <2 x i64> %num, <i64 4, i64 4>
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_v4i64(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %den_ptr = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i64 1
  %num = load <4 x i64>, <4 x i64> addrspace(1) * %in
  %den = load <4 x i64>, <4 x i64> addrspace(1) * %den_ptr
  %result = srem <4 x i64> %num, %den
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}

define amdgpu_kernel void @srem_v4i64_4(<4 x i64> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %num = load <4 x i64>, <4 x i64> addrspace(1) * %in
  %result = srem <4 x i64> %num, <i64 4, i64 4, i64 4, i64 4>
  store <4 x i64> %result, <4 x i64> addrspace(1)* %out
  ret void
}
