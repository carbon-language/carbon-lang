; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i1 @llvm.AMDGPU.class.f32(float, i32) #1
declare i1 @llvm.AMDGPU.class.f64(double, i32) #1
declare i32 @llvm.r600.read.tidig.x() #1
declare float @llvm.fabs.f32(float) #1
declare double @llvm.fabs.f64(double) #1

; SI-LABEL: {{^}}test_isinf_pattern:
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x204{{$}}
; SI: v_cmp_class_f32_e32 vcc, s{{[0-9]+}}, [[MASK]]
; SI-NOT: v_cmp
; SI: s_endpgm
define void @test_isinf_pattern(i32 addrspace(1)* nocapture %out, float %x) #0 {
  %fabs = tail call float @llvm.fabs.f32(float %x) #1
  %cmp = fcmp oeq float %fabs, 0x7FF0000000000000
  %ext = zext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_not_isinf_pattern_0:
; SI-NOT: v_cmp_class
; SI: s_endpgm
define void @test_not_isinf_pattern_0(i32 addrspace(1)* nocapture %out, float %x) #0 {
  %fabs = tail call float @llvm.fabs.f32(float %x) #1
  %cmp = fcmp ueq float %fabs, 0x7FF0000000000000
  %ext = zext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_not_isinf_pattern_1:
; SI-NOT: v_cmp_class
; SI: s_endpgm
define void @test_not_isinf_pattern_1(i32 addrspace(1)* nocapture %out, float %x) #0 {
  %fabs = tail call float @llvm.fabs.f32(float %x) #1
  %cmp = fcmp oeq float %fabs, 0xFFF0000000000000
  %ext = zext i1 %cmp to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
