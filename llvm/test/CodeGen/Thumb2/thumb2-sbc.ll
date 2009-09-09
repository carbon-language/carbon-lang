; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
; CHECK: f1:
; CHECK: subs r0, r0, r2
    %tmp = sub i64 %a, %b
    ret i64 %tmp
}
