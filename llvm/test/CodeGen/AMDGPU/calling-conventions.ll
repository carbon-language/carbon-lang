; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI %s

; Make sure we don't crash or assert on spir_kernel calling convention.

; SI-LABEL: {{^}}kernel:
; SI: s_endpgm
define spir_kernel void @kernel(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; FIXME: This is treated like a kernel
; SI-LABEL: {{^}}func:
; SI: s_endpgm
define spir_func void @func(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}
