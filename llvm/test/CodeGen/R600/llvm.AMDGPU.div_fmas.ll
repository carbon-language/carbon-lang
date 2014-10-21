; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare float @llvm.AMDGPU.div.fmas.f32(float, float, float, i1) nounwind readnone
declare double @llvm.AMDGPU.div.fmas.f64(double, double, double, i1) nounwind readnone

; SI-LABEL: {{^}}test_div_fmas_f32:
; SI-DAG: S_LOAD_DWORD [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: S_LOAD_DWORD [[SC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: S_LOAD_DWORD [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI-DAG: V_MOV_B32_e32 [[VC:v[0-9]+]], [[SC]]
; SI-DAG: V_MOV_B32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: V_DIV_FMAS_F32 [[RESULT:v[0-9]+]], [[SA]], [[VB]], [[VC]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
define void @test_div_fmas_f32(float addrspace(1)* %out, float %a, float %b, float %c, i1 %d) nounwind {
  %result = call float @llvm.AMDGPU.div.fmas.f32(float %a, float %b, float %c, i1 %d) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_fmas_f64:
; SI: V_DIV_FMAS_F64
define void @test_div_fmas_f64(double addrspace(1)* %out, double %a, double %b, double %c, i1 %d) nounwind {
  %result = call double @llvm.AMDGPU.div.fmas.f64(double %a, double %b, double %c, i1 %d) nounwind readnone
  store double %result, double addrspace(1)* %out, align 8
  ret void
}
