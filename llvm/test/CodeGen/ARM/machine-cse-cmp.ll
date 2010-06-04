; RUN: llc < %s -march=arm | FileCheck %s
;rdar://8003725

@G1 = external global i32
@G2 = external global i32

define i32 @f1(i32 %cond1, i32 %x1, i32 %x2, i32 %x3) {
entry:
; CHECK: cmp
; CHECK: moveq
; CHECK-NOT: cmp
; CHECK: moveq
    %tmp1 = icmp eq i32 %cond1, 0
    %tmp2 = select i1 %tmp1, i32 %x1, i32 %x2
    %tmp3 = select i1 %tmp1, i32 %x2, i32 %x3
    %tmp4 = add i32 %tmp2, %tmp3
    ret i32 %tmp4
}
