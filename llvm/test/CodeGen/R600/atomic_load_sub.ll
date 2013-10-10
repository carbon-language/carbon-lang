; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK-LABEL: @atomic_sub_local
; R600-CHECK: LDS_SUB *
; SI-CHECK-LABEL: @atomic_sub_local
; SI-CHECK: DS_SUB_U32_RTN 0
define void @atomic_sub_local(i32 addrspace(3)* %local) {
entry:
   %0 = atomicrmw volatile sub i32 addrspace(3)* %local, i32 5 seq_cst
   ret void
}

; R600-CHECK-LABEL: @atomic_sub_ret_local
; R600-CHECK: LDS_SUB_RET *
; SI-CHECK-LABEL: @atomic_sub_ret_local
; SI-CHECK: DS_SUB_U32_RTN 0
define void @atomic_sub_ret_local(i32 addrspace(1)* %out, i32 addrspace(3)* %local) {
entry:
  %0 = atomicrmw volatile sub i32 addrspace(3)* %local, i32 5 seq_cst
  store i32 %0, i32 addrspace(1)* %out
  ret void
}
