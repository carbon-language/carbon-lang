; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck -check-prefix CHECK-V7 %s
; RUN: llc -mtriple=thumbv8 %s -o - | FileCheck %s -check-prefix CHECK-V8
; PR11107

define i32 @test(i32 %a, i32 %b) {
entry:
 %cmp1 = icmp slt i32 %a, 0
 %sub1 = sub nsw i32 0, %a
 %abs1 = select i1 %cmp1, i32 %sub1, i32 %a
 %cmp2 = icmp slt i32 %b, 0
 %sub2 = sub nsw i32 0, %b
 %abs2 = select i1 %cmp2, i32 %sub2, i32 %b
 %add = add nsw i32 %abs1, %abs2
 ret i32 %add
}

; CHECK-V7:        cmp
; CHECK-V7-NEXT:   it    mi
; CHECK-V7-NEXT:   rsbmi
; CHECK-V7-NEXT:   cmp
; CHECK-V7-NEXT:   it    mi
; CHECK-V7-NEXT:   rsbmi

; CHECK-V8:        cmp
; CHECK-V8-NEXT:   bpl
; CHECK-V8:        rsbs
; CHECK-V8:        cmp
; CHECK-V8-NEXT:   bpl
; CHECK-V8:        rsbs

