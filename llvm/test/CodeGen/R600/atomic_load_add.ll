; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK-LABEL: @atomic_add_local
; R600-CHECK: LDS_ADD *
; SI-CHECK-LABEL: @atomic_add_local
; SI-CHECK: DS_ADD_U32_RTN
define void @atomic_add_local(i32 addrspace(3)* %local) {
entry:
   %0 = atomicrmw volatile add i32 addrspace(3)* %local, i32 5 seq_cst
   ret void
}

; R600-CHECK-LABEL: @atomic_add_ret_local
; R600-CHECK: LDS_ADD_RET *
; SI-CHECK-LABEL: @atomic_add_ret_local
; SI-CHECK: DS_ADD_U32_RTN
define void @atomic_add_ret_local(i32 addrspace(1)* %out, i32 addrspace(3)* %local) {
entry:
  %0 = atomicrmw volatile add i32 addrspace(3)* %local, i32 5 seq_cst
  store i32 %0, i32 addrspace(1)* %out
  ret void
}
