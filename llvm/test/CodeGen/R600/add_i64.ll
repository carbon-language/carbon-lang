; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s


declare i32 @llvm.r600.read.tidig.x() readnone

; SI-LABEL: {{^}}test_i64_vreg:
; SI: V_ADD_I32
; SI: V_ADDC_U32
define void @test_i64_vreg(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %inA, i64 addrspace(1)* noalias %inB) {
  %tid = call i32 @llvm.r600.read.tidig.x() readnone
  %a_ptr = getelementptr i64 addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr i64 addrspace(1)* %inB, i32 %tid
  %a = load i64 addrspace(1)* %a_ptr
  %b = load i64 addrspace(1)* %b_ptr
  %result = add i64 %a, %b
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; Check that the SGPR add operand is correctly moved to a VGPR.
; SI-LABEL: {{^}}sgpr_operand:
; SI: V_ADD_I32
; SI: V_ADDC_U32
define void @sgpr_operand(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in, i64 addrspace(1)* noalias %in_bar, i64 %a) {
  %foo = load i64 addrspace(1)* %in, align 8
  %result = add i64 %foo, %a
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; Swap the arguments. Check that the SGPR -> VGPR copy works with the
; SGPR as other operand.
;
; SI-LABEL: {{^}}sgpr_operand_reversed:
; SI: V_ADD_I32
; SI: V_ADDC_U32
define void @sgpr_operand_reversed(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in, i64 %a) {
  %foo = load i64 addrspace(1)* %in, align 8
  %result = add i64 %a, %foo
  store i64 %result, i64 addrspace(1)* %out
  ret void
}


; SI-LABEL: {{^}}test_v2i64_sreg:
; SI: S_ADD_U32
; SI: S_ADDC_U32
; SI: S_ADD_U32
; SI: S_ADDC_U32
define void @test_v2i64_sreg(<2 x i64> addrspace(1)* noalias %out, <2 x i64> %a, <2 x i64> %b) {
  %result = add <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}test_v2i64_vreg:
; SI: V_ADD_I32
; SI: V_ADDC_U32
; SI: V_ADD_I32
; SI: V_ADDC_U32
define void @test_v2i64_vreg(<2 x i64> addrspace(1)* noalias %out, <2 x i64> addrspace(1)* noalias %inA, <2 x i64> addrspace(1)* noalias %inB) {
  %tid = call i32 @llvm.r600.read.tidig.x() readnone
  %a_ptr = getelementptr <2 x i64> addrspace(1)* %inA, i32 %tid
  %b_ptr = getelementptr <2 x i64> addrspace(1)* %inB, i32 %tid
  %a = load <2 x i64> addrspace(1)* %a_ptr
  %b = load <2 x i64> addrspace(1)* %b_ptr
  %result = add <2 x i64> %a, %b
  store <2 x i64> %result, <2 x i64> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}trunc_i64_add_to_i32:
; SI: S_LOAD_DWORD s[[SREG0:[0-9]+]]
; SI: S_LOAD_DWORD s[[SREG1:[0-9]+]]
; SI: S_ADD_I32 [[SRESULT:s[0-9]+]], s[[SREG1]], s[[SREG0]]
; SI-NOT: ADDC
; SI: V_MOV_B32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: BUFFER_STORE_DWORD [[VRESULT]],
define void @trunc_i64_add_to_i32(i32 addrspace(1)* %out, i64 %a, i64 %b) {
  %add = add i64 %b, %a
  %trunc = trunc i64 %add to i32
  store i32 %trunc, i32 addrspace(1)* %out, align 8
  ret void
}
