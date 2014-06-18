; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.brev(i32) nounwind readnone

; FUNC-LABEL: @s_brev_i32:
; SI: S_LOAD_DWORD [[VAL:s[0-9]+]],
; SI: S_BREV_B32 [[SRESULT:s[0-9]+]], [[VAL]]
; SI: V_MOV_B32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: BUFFER_STORE_DWORD [[VRESULT]],
; SI: S_ENDPGM
define void @s_brev_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %ctlz = call i32 @llvm.AMDGPU.brev(i32 %val) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @v_brev_i32:
; SI: BUFFER_LOAD_DWORD [[VAL:v[0-9]+]],
; SI: V_BFREV_B32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: BUFFER_STORE_DWORD [[RESULT]],
; SI: S_ENDPGM
define void @v_brev_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32 addrspace(1)* %valptr, align 4
  %ctlz = call i32 @llvm.AMDGPU.brev(i32 %val) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}
