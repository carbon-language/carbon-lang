; RUN: llvm-as < %s | llc -march=thumb -mattr=+thumb2 | FileCheck %s

; 0x000000bb = 187
define i1 @f1(i32 %a) {
; CHECK: f1:
; CHECK: cmp r0, #187
    %tmp = icmp ne i32 %a, 187
    ret i1 %tmp
}

; 0x00aa00aa = 11141290
define i1 @f2(i32 %a) {
; CHECK: f2:
; CHECK: cmp.w r0, #11141290
    %tmp = icmp eq i32 %a, 11141290 
    ret i1 %tmp
}

; 0xcc00cc00 = 3422604288
define i1 @f3(i32 %a) {
; CHECK: f3:
; CHECK: cmp.w r0, #3422604288
    %tmp = icmp ne i32 %a, 3422604288
    ret i1 %tmp
}

; 0xdddddddd = 3722304989
define i1 @f4(i32 %a) {
; CHECK: f4:
; CHECK: cmp.w r0, #3722304989
    %tmp = icmp ne i32 %a, 3722304989
    ret i1 %tmp
}

; 0x00110000 = 1114112
define i1 @f5(i32 %a) {
; CHECK: f5:
; CHECK: cmp.w r0, #1114112
    %tmp = icmp eq i32 %a, 1114112
    ret i1 %tmp
}
