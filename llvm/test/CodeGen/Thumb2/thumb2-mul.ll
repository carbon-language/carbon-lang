; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK: f1:
; CHECK: muls r0, r1
    %tmp = mul i32 %a, %b
    ret i32 %tmp
}
