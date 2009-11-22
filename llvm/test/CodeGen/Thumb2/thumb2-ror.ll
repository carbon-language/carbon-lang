; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s


define i32 @f1(i32 %a) {
    %l8 = shl i32 %a, 10
    %r8 = lshr i32 %a, 22
    %tmp = or i32 %l8, %r8
    ret i32 %tmp
}
; CHECK: f1:
; CHECK: 	ror.w	r0, r0, #22
