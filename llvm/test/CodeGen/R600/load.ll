; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=R600-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck --check-prefix=R600-CHECK %s
; RUN: llc < %s -march=r600 -mcpu=SI | FileCheck --check-prefix=SI-CHECK  %s

; Load an i8 value from the global address space.
; R600-CHECK: @load_i8
; R600-CHECK: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}

; SI-CHECK: @load_i8
; SI-CHECK: BUFFER_LOAD_UBYTE VGPR{{[0-9]+}},
define void @load_i8(i32 addrspace(1)* %out, i8 addrspace(1)* %in) {
  %1 = load i8 addrspace(1)* %in
  %2 = zext i8 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; load an i32 value from the global address space.
; R600-CHECK: @load_i32
; R600-CHECK: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0

; SI-CHECK: @load_i32
; SI-CHECK: BUFFER_LOAD_DWORD VGPR{{[0-9]+}}
define void @load_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = load i32 addrspace(1)* %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; load a f32 value from the global address space.
; R600-CHECK: @load_f32
; R600-CHECK: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0

; SI-CHECK: @load_f32
; SI-CHECK: BUFFER_LOAD_DWORD VGPR{{[0-9]+}}
define void @load_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
entry:
  %0 = load float addrspace(1)* %in
  store float %0, float addrspace(1)* %out
  ret void
}

; Load an i32 value from the constant address space.
; R600-CHECK: @load_const_addrspace_i32
; R600-CHECK: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0

; SI-CHECK: @load_const_addrspace_i32
; SI-CHECK: S_LOAD_DWORD SGPR{{[0-9]+}}
define void @load_const_addrspace_i32(i32 addrspace(1)* %out, i32 addrspace(2)* %in) {
entry:
  %0 = load i32 addrspace(2)* %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; Load a f32 value from the constant address space.
; R600-CHECK: @load_const_addrspace_f32
; R600-CHECK: VTX_READ_32 T{{[0-9]+}}.X, T{{[0-9]+}}.X, 0

; SI-CHECK: @load_const_addrspace_f32
; SI-CHECK: S_LOAD_DWORD SGPR{{[0-9]+}}
define void @load_const_addrspace_f32(float addrspace(1)* %out, float addrspace(2)* %in) {
  %1 = load float addrspace(2)* %in
  store float %1, float addrspace(1)* %out
  ret void
}
