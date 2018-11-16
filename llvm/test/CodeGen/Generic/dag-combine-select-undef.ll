; RUN: llc < %s | FileCheck %s

define void @select_undef_n1(float addrspace(1)* %a, i32 %c) {
; CHECK-LABEL: select_undef_n1:
; CHECK:    movl $1065353216, (%rdi)
  %cc = icmp eq i32 %c, 0
  %sel = select i1 %cc, float 1.000000e+00, float undef
  store float %sel, float addrspace(1)* %a
  ret void
}

define void @select_undef_n2(float addrspace(1)* %a, i32 %c) {
; CHECK-LABEL: select_undef_n2:
; CHECK:    movl $1065353216, (%rdi)
  %cc = icmp eq i32 %c, 0
  %sel = select i1 %cc, float undef, float 1.000000e+00
  store float %sel, float addrspace(1)* %a
  ret void
}
