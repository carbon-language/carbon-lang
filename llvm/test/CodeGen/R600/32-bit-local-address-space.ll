; RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

; On Southern Islands GPUs the local address space(3) uses 32-bit pointers and
; the global address space(1) uses 64-bit pointers.  These tests check to make sure
; the correct pointer size is used for the local address space.

; The e{{32|64}} suffix on the instructions refers to the encoding size and not
; the size of the operands.  The operand size is denoted in the instruction name.
; Instructions with B32, U32, and I32 in their name take 32-bit operands, while
; instructions with B64, U64, and I64 take 64-bit operands.

; CHECK-LABEL: @local_address_load
; CHECK: V_MOV_B32_e{{32|64}} [[PTR:v[0-9]]]
; CHECK: DS_READ_B32 [[PTR]]
define void @local_address_load(i32 addrspace(1)* %out, i32 addrspace(3)* %in) {
entry:
  %0 = load i32 addrspace(3)* %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @local_address_gep
; CHECK: V_ADD_I32_e{{32|64}} [[PTR:v[0-9]]]
; CHECK: DS_READ_B32 [[PTR]]
define void @local_address_gep(i32 addrspace(1)* %out, i32 addrspace(3)* %in, i32 %offset) {
entry:
  %0 = getelementptr i32 addrspace(3)* %in, i32 %offset
  %1 = load i32 addrspace(3)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @local_address_gep_const_offset
; CHECK: V_ADD_I32_e{{32|64}} [[PTR:v[0-9]]]
; CHECK: DS_READ_B32 [[PTR]]
define void @local_address_gep_const_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %in) {
entry:
  %0 = getelementptr i32 addrspace(3)* %in, i32 1
  %1 = load i32 addrspace(3)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @null_32bit_lds_ptr:
; CHECK: V_CMP_NE_I32
; CHECK-NOT: V_CMP_NE_I32
; CHECK: V_CNDMASK_B32
define void @null_32bit_lds_ptr(i32 addrspace(1)* %out, i32 addrspace(3)* %lds) nounwind {
  %cmp = icmp ne i32 addrspace(3)* %lds, null
  %x = select i1 %cmp, i32 123, i32 456
  store i32 %x, i32 addrspace(1)* %out
  ret void
}