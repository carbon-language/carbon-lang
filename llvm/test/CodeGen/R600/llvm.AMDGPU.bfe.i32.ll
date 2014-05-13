; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -show-mc-encoding -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.bfe.i32(i32, i32, i32) nounwind readnone

; FUNC-LABEL: @bfe_i32_arg_arg_arg
; SI: V_BFE_I32
; EG: BFE_INT
; EG: encoding: [{{[x0-9a-f]+,[x0-9a-f]+,[x0-9a-f]+,[x0-9a-f]+,[x0-9a-f]+}},0xac
define void @bfe_i32_arg_arg_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 %src0, i32 %src1, i32 %src1) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_arg_arg_imm
; SI: V_BFE_I32
; EG: BFE_INT
define void @bfe_i32_arg_arg_imm(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 %src0, i32 %src1, i32 123) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_arg_imm_arg
; SI: V_BFE_I32
; EG: BFE_INT
define void @bfe_i32_arg_imm_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src2) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 %src0, i32 123, i32 %src2) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_imm_arg_arg
; SI: V_BFE_I32
; EG: BFE_INT
define void @bfe_i32_imm_arg_arg(i32 addrspace(1)* %out, i32 %src1, i32 %src2) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 123, i32 %src1, i32 %src2) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_bfe_print_arg
; SI: V_BFE_I32 v{{[0-9]+}}, v{{[0-9]+}}, 2, 8
define void @v_bfe_print_arg(i32 addrspace(1)* %out, i32 addrspace(1)* %src0) nounwind {
  %load = load i32 addrspace(1)* %src0, align 4
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 %load, i32 2, i32 8) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}
