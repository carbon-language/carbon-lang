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

; FUNC-LABEL: @bfe_i32_arg_0_width_reg_offset
; SI-NOT: BFE
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_arg_0_width_reg_offset(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.i32(i32 %src0, i32 %src1, i32 0) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_arg_0_width_imm_offset
; SI-NOT: BFE
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_arg_0_width_imm_offset(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfe_u32 = call i32 @llvm.AMDGPU.bfe.i32(i32 %src0, i32 8, i32 0) nounwind readnone
  store i32 %bfe_u32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_0
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_0(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 0, i32 0, i32 0) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_1
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_1(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 12334, i32 0, i32 0) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_2
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_2(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 0, i32 0, i32 1) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_3
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], -1
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_3(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 1, i32 0, i32 1) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_4
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], -1
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_4(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 4294967295, i32 0, i32 1) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_5
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], -1
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_5(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 128, i32 7, i32 1) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_6
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0xffffff80
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_6(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 128, i32 0, i32 8) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_7
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0x7f
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_7(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 127, i32 0, i32 8) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_8
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 1
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_8(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 127, i32 6, i32 8) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_9
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 1
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_9(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 65536, i32 16, i32 8) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_10
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_10(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 65535, i32 16, i32 16) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_11
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], -6
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_11(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 160, i32 4, i32 4) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_12
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_12(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 160, i32 31, i32 1) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_13
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 1
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_13(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 131070, i32 16, i32 16) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_14
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 40
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_14(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 160, i32 2, i32 30) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_15
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 10
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_15(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 160, i32 4, i32 28) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_16
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], -1
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_16(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 4294967295, i32 1, i32 7) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_17
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0x7f
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_17(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 255, i32 1, i32 31) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @bfe_i32_constant_fold_test_18
; SI-NOT: BFE
; SI: V_MOV_B32_e32 [[VREG:v[0-9]+]], 0
; SI: BUFFER_STORE_DWORD [[VREG]],
; SI: S_ENDPGM
; EG-NOT: BFE
define void @bfe_i32_constant_fold_test_18(i32 addrspace(1)* %out) nounwind {
  %bfe_i32 = call i32 @llvm.AMDGPU.bfe.i32(i32 255, i32 31, i32 1) nounwind readnone
  store i32 %bfe_i32, i32 addrspace(1)* %out, align 4
  ret void
}
