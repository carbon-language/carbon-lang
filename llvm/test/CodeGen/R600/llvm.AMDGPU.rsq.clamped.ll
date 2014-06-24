; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


declare float @llvm.AMDGPU.rsq.clamped.f32(float) nounwind readnone

; FUNC-LABEL: @rsq_clamped_f32
; SI: V_RSQ_CLAMP_F32_e32
; EG: RECIPSQRT_CLAMPED
define void @rsq_clamped_f32(float addrspace(1)* %out, float %src) nounwind {
  %rsq_clamped = call float @llvm.AMDGPU.rsq.clamped.f32(float %src) nounwind readnone
  store float %rsq_clamped, float addrspace(1)* %out, align 4
  ret void
}
