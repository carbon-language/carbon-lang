; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.AMDGPU.clamp.f32(float, float, float) nounwind readnone
declare float @llvm.AMDIL.clamp.f32(float, float, float) nounwind readnone

; FUNC-LABEL: @clamp_0_1_f32
; SI: S_LOAD_DWORD [[ARG:s[0-9]+]],
; SI: V_ADD_F32_e64 [[RESULT:v[0-9]+]], [[ARG]], 0, 1, 0
; SI: BUFFER_STORE_DWORD [[RESULT]]
; SI: S_ENDPGM

; EG: MOV_SAT
define void @clamp_0_1_f32(float addrspace(1)* %out, float %src) nounwind {
  %clamp = call float @llvm.AMDGPU.clamp.f32(float %src, float 0.0, float 1.0) nounwind readnone
  store float %clamp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @clamp_0_1_amdil_legacy_f32
; SI: S_LOAD_DWORD [[ARG:s[0-9]+]],
; SI: V_ADD_F32_e64 [[RESULT:v[0-9]+]], [[ARG]], 0, 1, 0
; SI: BUFFER_STORE_DWORD [[RESULT]]
define void @clamp_0_1_amdil_legacy_f32(float addrspace(1)* %out, float %src) nounwind {
  %clamp = call float @llvm.AMDIL.clamp.f32(float %src, float 0.0, float 1.0) nounwind readnone
  store float %clamp, float addrspace(1)* %out, align 4
  ret void
}
