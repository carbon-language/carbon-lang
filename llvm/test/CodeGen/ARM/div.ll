; RUN: llc < %s -march=arm | FileCheck %s -check-prefix=CHECK-ARM

define i32 @f1(i32 %a, i32 %b) {
entry:
; CHECK-ARM: f1
; CHECK-ARM: __divsi3
        %tmp1 = sdiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
entry:
; CHECK-ARM: f2
; CHECK-ARM: __udivsi3
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
entry:
; CHECK-ARM: f3
; CHECK-ARM: __modsi3
        %tmp1 = srem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
entry:
; CHECK-ARM: f4
; CHECK-ARM: __umodsi3
        %tmp1 = urem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

