; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; Test using an integer literal constant.
; Generated ASM should be:
; ADD_INT KC0[2].Z literal.x, 5
; or
; ADD_INT literal.x KC0[2].Z, 5

; CHECK: @i32_literal
; CHECK: ADD_INT * T{{[0-9]\.[XYZW]}}, KC0[2].Z, literal.x
; CHECK-NEXT: 5
define void @i32_literal(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0 = add i32 5, %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; Test using a float literal constant.
; Generated ASM should be:
; ADD KC0[2].Z literal.x, 5.0
; or
; ADD literal.x KC0[2].Z, 5.0

; CHECK: @float_literal
; CHECK: ADD * T{{[0-9]\.[XYZW]}}, KC0[2].Z, literal.x
; CHECK-NEXT: 1084227584(5.0
define void @float_literal(float addrspace(1)* %out, float %in) {
entry:
  %0 = fadd float 5.0, %in
  store float %0, float addrspace(1)* %out
  ret void
}
