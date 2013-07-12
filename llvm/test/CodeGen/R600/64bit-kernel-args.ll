; RUN: llc < %s -march=r600 -mcpu=tahiti | FileCheck %s

; SI-CHECK: @f64_kernel_arg
; SI-CHECK: BUFFER_STORE_DWORDX2
define void @f64_kernel_arg(double addrspace(1)* %out, double  %in) {
entry:
  store double %in, double addrspace(1)* %out
  ret void
}
