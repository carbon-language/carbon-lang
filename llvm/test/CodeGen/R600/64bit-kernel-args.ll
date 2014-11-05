; RUN: llc < %s -march=r600 -mcpu=tahiti -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; SI-CHECK: {{^}}f64_kernel_arg:
; SI-CHECK-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0x9
; SI-CHECK-DAG: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[0:1], 0xb
; SI-CHECK: buffer_store_dwordx2
define void @f64_kernel_arg(double addrspace(1)* %out, double  %in) {
entry:
  store double %in, double addrspace(1)* %out
  ret void
}
