; RUN: llc < %s -march=r600 -mcpu=tahiti -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; SI-CHECK: @f64_kernel_arg
; SI-CHECK-DAG: S_LOAD_DWORDX2 s[{{[0-9]:[0-9]}}], s[0:1], 9
; SI-CHECK-DAG: S_LOAD_DWORDX2 s[{{[0-9]:[0-9]}}], s[0:1], 11
; SI-CHECK: BUFFER_STORE_DWORDX2
define void @f64_kernel_arg(double addrspace(1)* %out, double  %in) {
entry:
  store double %in, double addrspace(1)* %out
  ret void
}
