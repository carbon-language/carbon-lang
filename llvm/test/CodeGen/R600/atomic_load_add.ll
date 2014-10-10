; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs < %s | FileCheck %s -check-prefix=SI -check-prefix=FUNC
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}atomic_add_local:
; R600: LDS_ADD *
; SI: DS_ADD_U32
define void @atomic_add_local(i32 addrspace(3)* %local) {
   %unused = atomicrmw volatile add i32 addrspace(3)* %local, i32 5 seq_cst
   ret void
}

; FUNC-LABEL: {{^}}atomic_add_local_const_offset:
; R600: LDS_ADD *
; SI: DS_ADD_U32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
define void @atomic_add_local_const_offset(i32 addrspace(3)* %local) {
  %gep = getelementptr i32 addrspace(3)* %local, i32 4
  %val = atomicrmw volatile add i32 addrspace(3)* %gep, i32 5 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_ret_local:
; R600: LDS_ADD_RET *
; SI: DS_ADD_RTN_U32
define void @atomic_add_ret_local(i32 addrspace(1)* %out, i32 addrspace(3)* %local) {
  %val = atomicrmw volatile add i32 addrspace(3)* %local, i32 5 seq_cst
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_ret_local_const_offset:
; R600: LDS_ADD_RET *
; SI: DS_ADD_RTN_U32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:20
define void @atomic_add_ret_local_const_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %local) {
  %gep = getelementptr i32 addrspace(3)* %local, i32 5
  %val = atomicrmw volatile add i32 addrspace(3)* %gep, i32 5 seq_cst
  store i32 %val, i32 addrspace(1)* %out
  ret void
}
