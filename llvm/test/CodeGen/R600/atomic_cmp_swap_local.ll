; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: @lds_atomic_cmpxchg_ret_i32_offset:
; SI: S_LOAD_DWORD [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: S_LOAD_DWORD [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: V_MOV_B32_e32 [[VCMP:v[0-9]+]], 7
; SI-DAG: V_MOV_B32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; SI-DAG: V_MOV_B32_e32 [[VSWAP:v[0-9]+]], [[SWAP]]
; SI: DS_CMPST_RTN_B32 [[RESULT:v[0-9]+]], [[VPTR]], [[VCMP]], [[VSWAP]], 0x10, [M0]
; SI: S_ENDPGM
define void @lds_atomic_cmpxchg_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr, i32 %swap) nounwind {
  %gep = getelementptr i32 addrspace(3)* %ptr, i32 4
  %result = cmpxchg i32 addrspace(3)* %gep, i32 7, i32 %swap seq_cst monotonic
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}
