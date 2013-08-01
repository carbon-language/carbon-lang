; RUN: llc < %s -march=r600 -mcpu=tahiti | FileCheck %s --check-prefix=SI-CHECK

; SI-CHECK: @f64_kernel_arg
; SI-CHECK-DAG: S_LOAD_DWORDX2 SGPR{{[0-9]}}_SGPR{{[0-9]}}, SGPR0_SGPR1, 9
; SI-CHECK-DAG: S_LOAD_DWORDX2 SGPR{{[0-9]}}_SGPR{{[0-9]}}, SGPR0_SGPR1, 11
; SI-CHECK: BUFFER_STORE_DWORDX2
define void @f64_kernel_arg(double addrspace(1)* %out, double  %in) {
entry:
  store double %in, double addrspace(1)* %out
  ret void
}
