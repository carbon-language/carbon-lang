; XFAIL: *
; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare float @llvm.AMDGPU.div.scale.f32(float, float) nounwind readnone
declare double @llvm.AMDGPU.div.scale.f64(double, double) nounwind readnone

; SI-LABEL @test_div_scale_f32:
define void @test_div_scale_f32(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr) nounwind {
  %a = load float addrspace(1)* %aptr, align 4
  %b = load float addrspace(1)* %bptr, align 4
  %result = call float @llvm.AMDGPU.div.scale.f32(float %a, float %b) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL @test_div_scale_f64:
define void @test_div_scale_f64(double addrspace(1)* %out, double addrspace(1)* %aptr, double addrspace(1)* %bptr) nounwind {
  %a = load double addrspace(1)* %aptr, align 8
  %b = load double addrspace(1)* %bptr, align 8
  %result = call double @llvm.AMDGPU.div.scale.f64(double %a, double %b) nounwind readnone
  store double %result, double addrspace(1)* %out, align 8
  ret void
}
