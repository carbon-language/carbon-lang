; RUN: llc -march=r600 -mcpu=SI -mattr=-fp32-denormals -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=SI-UNSAFE -check-prefix=SI %s
; RUN: llc -march=r600 -mcpu=SI -mattr=-fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI %s

declare float @llvm.sqrt.f32(float) nounwind readnone
declare double @llvm.sqrt.f64(double) nounwind readnone

; SI-LABEL: {{^}}rsq_f32:
; SI: V_RSQ_F32_e32
; SI: S_ENDPGM
define void @rsq_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) nounwind {
  %val = load float addrspace(1)* %in, align 4
  %sqrt = call float @llvm.sqrt.f32(float %val) nounwind readnone
  %div = fdiv float 1.0, %sqrt
  store float %div, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}rsq_f64:
; SI-UNSAFE: V_RSQ_F64_e32
; SI-SAFE: V_SQRT_F64_e32
; SI: S_ENDPGM
define void @rsq_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) nounwind {
  %val = load double addrspace(1)* %in, align 4
  %sqrt = call double @llvm.sqrt.f64(double %val) nounwind readnone
  %div = fdiv double 1.0, %sqrt
  store double %div, double addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}rsq_f32_sgpr:
; SI: V_RSQ_F32_e32 {{v[0-9]+}}, {{s[0-9]+}}
; SI: S_ENDPGM
define void @rsq_f32_sgpr(float addrspace(1)* noalias %out, float %val) nounwind {
  %sqrt = call float @llvm.sqrt.f32(float %val) nounwind readnone
  %div = fdiv float 1.0, %sqrt
  store float %div, float addrspace(1)* %out, align 4
  ret void
}
