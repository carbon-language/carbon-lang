; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-a8    | FileCheck %s -check-prefix=CHECK-SWDIV
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=swift        | FileCheck %s -check-prefix=CHECK-HWDIV
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-r4    | FileCheck %s -check-prefix=CHECK-SWDIV
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-r4f   | FileCheck %s -check-prefix=CHECK-SWDIV
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-r5    | FileCheck %s -check-prefix=CHECK-HWDIV

define i32 @f1(i32 %a, i32 %b) {
entry:
; CHECK-SWDIV: f1
; CHECK-SWDIV: __divsi3

; CHECK-HWDIV: f1
; CHECK-HWDIV: sdiv
        %tmp1 = sdiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
entry:
; CHECK-SWDIV: f2
; CHECK-SWDIV: __udivsi3

; CHECK-HWDIV: f2
; CHECK-HWDIV: udiv
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
entry:
; CHECK-SWDIV: f3
; CHECK-SWDIV: __modsi3

; CHECK-HWDIV: f3
; CHECK-HWDIV: sdiv
; CHECK-HWDIV: mls
        %tmp1 = srem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
entry:
; CHECK-SWDIV: f4
; CHECK-SWDIV: __umodsi3

; CHECK-HWDIV: f4
; CHECK-HWDIV: udiv
; CHECK-HWDIV: mls
        %tmp1 = urem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

