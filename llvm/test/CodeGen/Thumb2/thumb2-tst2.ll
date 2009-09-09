; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i1 @f1(i32 %a, i32 %b) {
; CHECK: f1:
; CHECK: tst r0, r1
    %tmp = and i32 %a, %b
    %tmp1 = icmp ne i32 %tmp, 0
    ret i1 %tmp1
}

define i1 @f2(i32 %a, i32 %b) {
; CHECK: f2:
; CHECK: tst r0, r1
    %tmp = and i32 %a, %b
    %tmp1 = icmp eq i32 %tmp, 0
    ret i1 %tmp1
}

define i1 @f3(i32 %a, i32 %b) {
; CHECK: f3:
; CHECK: tst r0, r1
    %tmp = and i32 %a, %b
    %tmp1 = icmp ne i32 0, %tmp
    ret i1 %tmp1
}

define i1 @f4(i32 %a, i32 %b) {
; CHECK: f4:
; CHECK: tst r0, r1
    %tmp = and i32 %a, %b
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}

define i1 @f6(i32 %a, i32 %b) {
; CHECK: f6:
; CHECK: tst.w r0, r1, lsl #5
    %tmp = shl i32 %b, 5
    %tmp1 = and i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    ret i1 %tmp2
}

define i1 @f7(i32 %a, i32 %b) {
; CHECK: f7:
; CHECK: tst.w r0, r1, lsr #6
    %tmp = lshr i32 %b, 6
    %tmp1 = and i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    ret i1 %tmp2
}

define i1 @f8(i32 %a, i32 %b) {
; CHECK: f8:
; CHECK: tst.w r0, r1, asr #7
    %tmp = ashr i32 %b, 7
    %tmp1 = and i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    ret i1 %tmp2
}

define i1 @f9(i32 %a, i32 %b) {
; CHECK: f9:
; CHECK: tst.w r0, r0, ror #8
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = and i32 %a, %tmp
    %tmp2 = icmp eq i32 %tmp1, 0
    ret i1 %tmp2
}
