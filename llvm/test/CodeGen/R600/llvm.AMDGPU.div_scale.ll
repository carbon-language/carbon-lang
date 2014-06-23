; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare { float, i1 } @llvm.AMDGPU.div.scale.f32(float, float, i1) nounwind readnone
declare { double, i1 } @llvm.AMDGPU.div.scale.f64(double, double, i1) nounwind readnone

; SI-LABEL @test_div_scale_f32_1:
; SI: V_DIV_SCALE_F32
define void @test_div_scale_f32_1(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr) nounwind {
  %a = load float addrspace(1)* %aptr, align 4
  %b = load float addrspace(1)* %bptr, align 4
  %result = call { float, i1 } @llvm.AMDGPU.div.scale.f32(float %a, float %b, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL @test_div_scale_f32_2:
; SI: V_DIV_SCALE_F32
define void @test_div_scale_f32_2(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr) nounwind {
  %a = load float addrspace(1)* %aptr, align 4
  %b = load float addrspace(1)* %bptr, align 4
  %result = call { float, i1 } @llvm.AMDGPU.div.scale.f32(float %a, float %b, i1 true) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL @test_div_scale_f64_1:
; SI: V_DIV_SCALE_F64
define void @test_div_scale_f64_1(double addrspace(1)* %out, double addrspace(1)* %aptr, double addrspace(1)* %bptr, double addrspace(1)* %cptr) nounwind {
  %a = load double addrspace(1)* %aptr, align 8
  %b = load double addrspace(1)* %bptr, align 8
  %result = call { double, i1 } @llvm.AMDGPU.div.scale.f64(double %a, double %b, i1 false) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL @test_div_scale_f64_1:
; SI: V_DIV_SCALE_F64
define void @test_div_scale_f64_2(double addrspace(1)* %out, double addrspace(1)* %aptr, double addrspace(1)* %bptr, double addrspace(1)* %cptr) nounwind {
  %a = load double addrspace(1)* %aptr, align 8
  %b = load double addrspace(1)* %bptr, align 8
  %result = call { double, i1 } @llvm.AMDGPU.div.scale.f64(double %a, double %b, i1 true) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}
