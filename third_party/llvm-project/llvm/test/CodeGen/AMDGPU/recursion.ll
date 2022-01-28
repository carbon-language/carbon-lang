; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}recursive:
; CHECK: ScratchSize: 16
define void @recursive() {
  call void @recursive()
  store volatile i32 0, i32 addrspace(1)* undef
  ret void
}

; CHECK-LABEL: {{^}}tail_recursive:
; CHECK: ScratchSize: 0
define void @tail_recursive() {
  tail call void @tail_recursive()
  ret void
}

define void @calls_tail_recursive() norecurse {
  tail call void @tail_recursive()
  ret void
}

; CHECK-LABEL: {{^}}tail_recursive_with_stack:
define void @tail_recursive_with_stack() {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  tail call void @tail_recursive_with_stack()
  ret void
}

; For an arbitrary recursive call, report a large number for unknown stack usage.
; CHECK-LABEL: {{^}}calls_recursive:
; CHECK: .amdhsa_private_segment_fixed_size 16400{{$}}
define amdgpu_kernel void @calls_recursive() {
  call void @recursive()
  ret void
}

; Make sure we do not report a huge stack size for tail recursive
; functions
; CHECK-LABEL: {{^}}kernel_indirectly_calls_tail_recursive:
; CHECK: .amdhsa_private_segment_fixed_size 0{{$}}
define amdgpu_kernel void @kernel_indirectly_calls_tail_recursive() {
  call void @calls_tail_recursive()
  ret void
}

; TODO: Even though tail_recursive is only called as a tail call, we
; end up treating it as generally recursive call from the regular call
; in the kernel.

; CHECK-LABEL: {{^}}kernel_calls_tail_recursive:
; CHECK: .amdhsa_private_segment_fixed_size 16384{{$}}
define amdgpu_kernel void @kernel_calls_tail_recursive() {
  call void @tail_recursive()
  ret void
}

; CHECK-LABEL: {{^}}kernel_calls_tail_recursive_with_stack:
; CHECK: .amdhsa_private_segment_fixed_size 16384{{$}}
define amdgpu_kernel void @kernel_calls_tail_recursive_with_stack() {
  call void @tail_recursive_with_stack()
  ret void
}
