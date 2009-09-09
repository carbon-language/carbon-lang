; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; 171 = 0x000000ab
define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: movs r0, #171
    %tmp = add i32 0, 171
    ret i32 %tmp
}

; 1179666 = 0x00120012
define i32 @f2(i32 %a) {
; CHECK: f2:
; CHECK: mov.w r0, #1179666
    %tmp = add i32 0, 1179666
    ret i32 %tmp
}

; 872428544 = 0x34003400
define i32 @f3(i32 %a) {
; CHECK: f3:
; CHECK: mov.w r0, #872428544
    %tmp = add i32 0, 872428544
    ret i32 %tmp
}

; 1448498774 = 0x56565656
define i32 @f4(i32 %a) {
; CHECK: f4:
; CHECK: mov.w r0, #1448498774
    %tmp = add i32 0, 1448498774
    ret i32 %tmp
}

; 66846720 = 0x03fc0000
define i32 @f5(i32 %a) {
; CHECK: f5:
; CHECK: mov.w r0, #66846720
    %tmp = add i32 0, 66846720
    ret i32 %tmp
}
