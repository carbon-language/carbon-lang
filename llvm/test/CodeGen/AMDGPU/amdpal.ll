; RUN: llc < %s -mtriple=amdgcn--amdpal -mcpu=tahiti | FileCheck --check-prefix=PAL %s

; PAL: .AMDGPU.config

define amdgpu_kernel void @simple(i32 addrspace(1)* %out) {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

