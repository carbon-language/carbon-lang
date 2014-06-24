; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.AMDGPU.rsq.f32(float) nounwind readnone

; FUNC-LABEL: @rsq_f32
; SI: V_RSQ_F32_e32
; EG: RECIPSQRT_IEEE
define void @rsq_f32(float addrspace(1)* %out, float %src) nounwind {
  %rsq = call float @llvm.AMDGPU.rsq.f32(float %src) nounwind readnone
  store float %rsq, float addrspace(1)* %out, align 4
  ret void
}
