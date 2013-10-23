; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG %s


define void @trunc_i64_to_i32_store(i32 addrspace(1)* %out, i64 %in) {
; SI-LABEL: @trunc_i64_to_i32_store
; SI: S_LOAD_DWORD SGPR0, SGPR0_SGPR1, 11
; SI: V_MOV_B32_e32 VGPR0, SGPR0
; SI: BUFFER_STORE_DWORD VGPR0

; EG-LABEL: @trunc_i64_to_i32_store
; EG: MEM_RAT_CACHELESS STORE_RAW T0.X, T1.X, 1
; EG: LSHR
; EG-NEXT: 2(

  %result = trunc i64 %in to i32 store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @trunc_shl_i64:
; SI: S_LOAD_DWORDX2
; SI: S_LOAD_DWORDX2 [[SREG:SGPR[0-9]+_SGPR[0-9]+]]
; SI: V_LSHL_B64 [[LO_VREG:VGPR[0-9]+]]_VGPR{{[0-9]+}}, [[SREG]], 2
; SI: BUFFER_STORE_DWORD [[LO_VREG]],
define void @trunc_shl_i64(i32 addrspace(1)* %out, i64 %a) {
  %b = shl i64 %a, 2
  %result = trunc i64 %b to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}
