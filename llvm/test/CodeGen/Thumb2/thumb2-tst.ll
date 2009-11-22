; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s


; 0x000000bb = 187
define i1 @f1(i32 %a) {
    %tmp = and i32 %a, 187
    %tmp1 = icmp ne i32 %tmp, 0
    ret i1 %tmp1
}
; CHECK: f1:
; CHECK: 	tst.w	r0, #187

; 0x000000bb = 187
define i1 @f2(i32 %a) {
    %tmp = and i32 %a, 187
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}
; CHECK: f2:
; CHECK: 	tst.w	r0, #187

; 0x00aa00aa = 11141290
define i1 @f3(i32 %a) {
    %tmp = and i32 %a, 11141290 
    %tmp1 = icmp eq i32 %tmp, 0
    ret i1 %tmp1
}
; CHECK: f3:
; CHECK: 	tst.w	r0, #11141290

; 0x00aa00aa = 11141290
define i1 @f4(i32 %a) {
    %tmp = and i32 %a, 11141290 
    %tmp1 = icmp ne i32 0, %tmp
    ret i1 %tmp1
}
; CHECK: f4:
; CHECK: 	tst.w	r0, #11141290

; 0xcc00cc00 = 3422604288
define i1 @f5(i32 %a) {
    %tmp = and i32 %a, 3422604288
    %tmp1 = icmp ne i32 %tmp, 0
    ret i1 %tmp1
}
; CHECK: f5:
; CHECK: 	tst.w	r0, #-872363008

; 0xcc00cc00 = 3422604288
define i1 @f6(i32 %a) {
    %tmp = and i32 %a, 3422604288
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}
; CHECK: f6:
; CHECK: 	tst.w	r0, #-872363008

; 0xdddddddd = 3722304989
define i1 @f7(i32 %a) {
    %tmp = and i32 %a, 3722304989
    %tmp1 = icmp eq i32 %tmp, 0
    ret i1 %tmp1
}
; CHECK: f7:
; CHECK: 	tst.w	r0, #-572662307

; 0xdddddddd = 3722304989
define i1 @f8(i32 %a) {
    %tmp = and i32 %a, 3722304989
    %tmp1 = icmp ne i32 0, %tmp
    ret i1 %tmp1
}
; CHECK: f8:
; CHECK: 	tst.w	r0, #-572662307

; 0x00110000 = 1114112
define i1 @f9(i32 %a) {
    %tmp = and i32 %a, 1114112
    %tmp1 = icmp ne i32 %tmp, 0
    ret i1 %tmp1
}
; CHECK: f9:
; CHECK: 	tst.w	r0, #1114112

; 0x00110000 = 1114112
define i1 @f10(i32 %a) {
    %tmp = and i32 %a, 1114112
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}
; CHECK: f10:
; CHECK: 	tst.w	r0, #1114112
