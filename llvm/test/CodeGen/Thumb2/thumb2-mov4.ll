; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f6(i32 %a) {
;CHECK: f6
;CHECK: movw    r0, #65535
    %tmp = add i32 0, 65535
    ret i32 %tmp
}
