; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG %s

define void @trunc_i64_to_i32_store(i32 addrspace(1)* %out, i64 %in) {
; SI-LABEL: @trunc_i64_to_i32_store
; SI: S_LOAD_DWORD s0, s[0:1], 0xb
; SI: V_MOV_B32_e32 v0, s0
; SI: BUFFER_STORE_DWORD v0

; EG-LABEL: @trunc_i64_to_i32_store
; EG: MEM_RAT_CACHELESS STORE_RAW T0.X, T1.X, 1
; EG: LSHR
; EG-NEXT: 2(

  %result = trunc i64 %in to i32 store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @trunc_load_shl_i64:
; SI-DAG: S_LOAD_DWORDX2
; SI-DAG: S_LOAD_DWORD [[SREG:s[0-9]+]],
; SI: S_LSHL_B32 [[SHL:s[0-9]+]], [[SREG]], 2
; SI: V_MOV_B32_e32 [[VSHL:v[0-9]+]], [[SHL]]
; SI: BUFFER_STORE_DWORD [[VSHL]],
define void @trunc_load_shl_i64(i32 addrspace(1)* %out, i64 %a) {
  %b = shl i64 %a, 2
  %result = trunc i64 %b to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: @trunc_shl_i64:
; SI: S_LOAD_DWORDX2 s{{\[}}[[LO_SREG:[0-9]+]]:{{[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI: S_ADD_I32 s[[LO_SREG2:[0-9]+]], s[[LO_SREG]],
; SI: S_ADDC_U32
; SI: S_LSHL_B64 s{{\[}}[[LO_SHL:[0-9]+]]:{{[0-9]+\]}}, s{{\[}}[[LO_SREG2]]:{{[0-9]+\]}}, 2
; SI: V_MOV_B32_e32 v[[LO_VREG:[0-9]+]], s[[LO_SHL]]
; SI: BUFFER_STORE_DWORD v[[LO_VREG]],
define void @trunc_shl_i64(i64 addrspace(1)* %out2, i32 addrspace(1)* %out, i64 %a) {
  %aa = add i64 %a, 234 ; Prevent shrinking store.
  %b = shl i64 %aa, 2
  %result = trunc i64 %b to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  store i64 %b, i64 addrspace(1)* %out2, align 8 ; Prevent reducing ops to 32-bits
  ret void
}

; SI-LABEL: @trunc_i32_to_i1:
; SI: V_AND_B32
; SI: V_CMP_EQ_I32
define void @trunc_i32_to_i1(i32 addrspace(1)* %out, i32 %a) {
  %trunc = trunc i32 %a to i1
  %result = select i1 %trunc, i32 1, i32 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}
