; RUN: llc -march=r600 -mcpu=SI -mattr=-fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=SI-UNSAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=SI -mattr=-fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s

; XUN: llc -march=r600 -mcpu=SI -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE-SPDENORM -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.AMDGPU.rcp.f32(float) nounwind readnone
declare double @llvm.AMDGPU.rcp.f64(double) nounwind readnone


declare float @llvm.sqrt.f32(float) nounwind readnone
declare double @llvm.sqrt.f64(double) nounwind readnone

; FUNC-LABEL: @rcp_f32
; SI: V_RCP_F32_e32
define void @rcp_f32(float addrspace(1)* %out, float %src) nounwind {
  %rcp = call float @llvm.AMDGPU.rcp.f32(float %src) nounwind readnone
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @rcp_f64
; SI: V_RCP_F64_e32
define void @rcp_f64(double addrspace(1)* %out, double %src) nounwind {
  %rcp = call double @llvm.AMDGPU.rcp.f64(double %src) nounwind readnone
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @rcp_pat_f32
; SI-SAFE: V_RCP_F32_e32
; XSI-SAFE-SPDENORM-NOT: V_RCP_F32_e32
define void @rcp_pat_f32(float addrspace(1)* %out, float %src) nounwind {
  %rcp = fdiv float 1.0, %src
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @rcp_pat_f64
; SI: V_RCP_F64_e32
define void @rcp_pat_f64(double addrspace(1)* %out, double %src) nounwind {
  %rcp = fdiv double 1.0, %src
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @rsq_rcp_pat_f32
; SI-UNSAFE: V_RSQ_F32_e32
; SI-SAFE: V_SQRT_F32_e32
; SI-SAFE: V_RCP_F32_e32
define void @rsq_rcp_pat_f32(float addrspace(1)* %out, float %src) nounwind {
  %sqrt = call float @llvm.sqrt.f32(float %src) nounwind readnone
  %rcp = call float @llvm.AMDGPU.rcp.f32(float %sqrt) nounwind readnone
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @rsq_rcp_pat_f64
; SI-UNSAFE: V_RSQ_F64_e32
; SI-SAFE-NOT: V_RSQ_F64_e32
define void @rsq_rcp_pat_f64(double addrspace(1)* %out, double %src) nounwind {
  %sqrt = call double @llvm.sqrt.f64(double %src) nounwind readnone
  %rcp = call double @llvm.AMDGPU.rcp.f64(double %sqrt) nounwind readnone
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}
