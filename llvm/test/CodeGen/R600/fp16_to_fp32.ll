; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i16 @llvm.convert.to.fp16(float) nounwind readnone

; SI-LABEL: @test_convert_fp16_to_fp32:
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]]
; SI: V_CVT_F16_F32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_SHORT [[RESULT]]
define void @test_convert_fp16_to_fp32(i16 addrspace(1)* noalias %out, float addrspace(1)* noalias %in) nounwind {
  %val = load float addrspace(1)* %in, align 4
  %cvt = call i16 @llvm.convert.to.fp16(float %val) nounwind readnone
  store i16 %cvt, i16 addrspace(1)* %out, align 2
  ret void
}
