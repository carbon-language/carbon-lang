; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; Load an i8 value from the global address space.
; CHECK: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}

define void @load_i8(i32 addrspace(1)* %out, i8 addrspace(1)* %in) {
  %1 = load i8 addrspace(1)* %in
  %2 = zext i8 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; Load a f32 value from the constant address space.
; CHECK: VTX_READ_32 T{{[0-9]+\.X, T[0-9]+\.X}}

define void @load_const_addrspace_f32(float addrspace(1)* %out, float addrspace(2)* %in) {
  %1 = load float addrspace(2)* %in
  store float %1, float addrspace(1)* %out
  ret void
}
