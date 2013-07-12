; RUN: llc < %s -march=r600 -mcpu=tahiti | FileCheck %s

; load a f64 value from the global address space.
; CHECK: @load_f64
; CHECK: BUFFER_LOAD_DWORDX2 VGPR{{[0-9]+}}
define void @load_f64(double addrspace(1)* %out, double addrspace(1)* %in) {
entry:
  %0 = load double addrspace(1)* %in
  store double %0, double addrspace(1)* %out
  ret void
}

; Load a f64 value from the constant address space.
; CHECK: @load_const_addrspace_f64
; CHECK: S_LOAD_DWORDX2 SGPR{{[0-9]+}}
define void @load_const_addrspace_f64(double addrspace(1)* %out, double addrspace(2)* %in) {
  %1 = load double addrspace(2)* %in
  store double %1, double addrspace(1)* %out
  ret void
}
