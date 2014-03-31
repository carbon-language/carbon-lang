; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.bfe.u32(i32, i32, i32) nounwind readnone

; FUNC-LABEL: @bfe_u32_arg_arg_arg
; SI: V_BFE_U32
; EG: BFE_UINT
define void @bfe_u32_arg_arg_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %src0, i32 %src1, i32 %src1) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_u32_arg_arg_imm
; SI: V_BFE_U32
; EG: BFE_UINT
define void @bfe_u32_arg_arg_imm(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %src0, i32 %src1, i32 123) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_u32_arg_imm_arg
; SI: V_BFE_U32
; EG: BFE_UINT
define void @bfe_u32_arg_imm_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src2) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 %src0, i32 123, i32 %src2) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_u32_imm_arg_arg
; SI: V_BFE_U32
; EG: BFE_UINT
define void @bfe_u32_imm_arg_arg(i32 addrspace(1)* %out, i32 %src1, i32 %src2) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.u32(i32 123, i32 %src1, i32 %src2) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}
