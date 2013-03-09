; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; Test using an integer literal constant.
; Generated ASM should be:
; ADD_INT REG literal.x, 5
; or
; ADD_INT literal.x REG, 5

; CHECK: @i32_literal
; CHECK: ADD_INT {{[A-Z0-9,. ]*}}literal.x,{{[A-Z0-9,. ]*}} 5
define void @i32_literal(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0 = add i32 5, %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; Test using a float literal constant.
; Generated ASM should be:
; ADD REG literal.x, 5.0
; or
; ADD literal.x REG, 5.0

; CHECK: @float_literal
; CHECK: ADD {{[A-Z0-9,. ]*}}literal.x,{{[A-Z0-9,. ]*}} {{[0-9]+}}(5.0
define void @float_literal(float addrspace(1)* %out, float %in) {
entry:
  %0 = fadd float 5.0, %in
  store float %0, float addrspace(1)* %out
  ret void
}

