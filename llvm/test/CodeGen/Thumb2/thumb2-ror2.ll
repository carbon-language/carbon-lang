; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a, i32 %b) {
; CHECK: f1:
; CHECK: rors r0, r1
    %db = sub i32 32, %b
    %l8 = shl i32 %a, %b
    %r8 = lshr i32 %a, %db
    %tmp = or i32 %l8, %r8
    ret i32 %tmp
}
