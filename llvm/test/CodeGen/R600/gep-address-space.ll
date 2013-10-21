; RUN: llc -march=r600 -mcpu=SI < %s | FileCheck %s

define void @use_gep_address_space([1024 x i32] addrspace(3)* %array) nounwind {
; CHECK-LABEL @use_gep_address_space:
; CHECK: ADD_I32
  %p = getelementptr [1024 x i32] addrspace(3)* %array, i16 0, i16 16
  store i32 99, i32 addrspace(3)* %p
  ret void
}

