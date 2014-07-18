; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare float @llvm.convert.from.fp16.f32(i16) nounwind readnone
declare double @llvm.convert.from.fp16.f64(i16) nounwind readnone

; SI-LABEL: @test_convert_fp16_to_fp32:
; SI: BUFFER_LOAD_USHORT [[VAL:v[0-9]+]]
; SI: V_CVT_F32_F16_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[RESULT]]
define void @test_convert_fp16_to_fp32(float addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in) nounwind {
  %val = load i16 addrspace(1)* %in, align 2
  %cvt = call float @llvm.convert.from.fp16.f32(i16 %val) nounwind readnone
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}


; SI-LABEL: @test_convert_fp16_to_fp64:
; SI: BUFFER_LOAD_USHORT [[VAL:v[0-9]+]]
; SI: V_CVT_F32_F16_e32 [[RESULT32:v[0-9]+]], [[VAL]]
; SI: V_CVT_F64_F32_e32 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[RESULT32]]
; SI: BUFFER_STORE_DWORDX2 [[RESULT]]
define void @test_convert_fp16_to_fp64(double addrspace(1)* noalias %out, i16 addrspace(1)* noalias %in) nounwind {
  %val = load i16 addrspace(1)* %in, align 2
  %cvt = call double @llvm.convert.from.fp16.f64(i16 %val) nounwind readnone
  store double %cvt, double addrspace(1)* %out, align 4
  ret void
}
