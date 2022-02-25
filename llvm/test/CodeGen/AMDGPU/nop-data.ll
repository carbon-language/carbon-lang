; RUN: llc -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mcpu=fiji -filetype=obj < %s | llvm-objdump -d - --mcpu=fiji | FileCheck %s

; CHECK: <kernel0>:
; CHECK: s_endpgm
define amdgpu_kernel void @kernel0() align 256 {
entry:
  ret void
}

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0

; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0
; CHECK-NEXT: s_nop 0  // 0000000001FC: BF800000

; CHECK-EMPTY:
; CHECK-NEXT: <kernel1>:
; CHECK: s_endpgm
define amdgpu_kernel void @kernel1(i32 addrspace(1)* addrspace(4)* %ptr.out) align 256 {
entry:
  ret void
}
