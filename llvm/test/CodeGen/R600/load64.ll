; RUN: llc < %s -march=r600 -mcpu=tahiti -verify-machineinstrs | FileCheck %s

; load a f64 value from the global address space.
; CHECK-LABEL: @load_f64:
; CHECK: BUFFER_LOAD_DWORDX2 v[{{[0-9]+:[0-9]+}}]
; CHECK: BUFFER_STORE_DWORDX2 v[{{[0-9]+:[0-9]+}}]
define void @load_f64(double addrspace(1)* %out, double addrspace(1)* %in) {
  %1 = load double addrspace(1)* %in
  store double %1, double addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @load_i64:
; CHECK: BUFFER_LOAD_DWORDX2 v[{{[0-9]+:[0-9]+}}]
; CHECK: BUFFER_STORE_DWORDX2 v[{{[0-9]+:[0-9]+}}]
define void @load_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tmp = load i64 addrspace(1)* %in
  store i64 %tmp, i64 addrspace(1)* %out, align 8
  ret void
}

; Load a f64 value from the constant address space.
; CHECK-LABEL: @load_const_addrspace_f64:
; CHECK: S_LOAD_DWORDX2 s[{{[0-9]+:[0-9]+}}]
; CHECK: BUFFER_STORE_DWORDX2 v[{{[0-9]+:[0-9]+}}]
define void @load_const_addrspace_f64(double addrspace(1)* %out, double addrspace(2)* %in) {
  %1 = load double addrspace(2)* %in
  store double %1, double addrspace(1)* %out
  ret void
}
