; RUN: opt < %s -instsimplify -S | FileCheck %s

define i32 @select1(i32 %x, i1 %b) {
; CHECK-LABEL: @select1(
  %rhs = select i1 %b, i32 %x, i32 1
  %rem = srem i32 %x, %rhs
  ret i32 %rem
; CHECK: ret i32 0
}

define i32 @select2(i32 %x, i1 %b) {
; CHECK-LABEL: @select2(
  %rhs = select i1 %b, i32 %x, i32 1
  %rem = urem i32 %x, %rhs
  ret i32 %rem
; CHECK: ret i32 0
}

define i32 @rem1(i32 %x, i32 %n) {
; CHECK-LABEL: @rem1(
; CHECK-NEXT: %mod = srem i32 %x, %n
; CHECK-NEXT: ret i32 %mod
 %mod = srem i32 %x, %n
 %mod1 = srem i32 %mod, %n
 ret i32 %mod1
}

define i32 @rem2(i32 %x, i32 %n) {
; CHECK-LABEL: @rem2(
; CHECK-NEXT: %mod = urem i32 %x, %n
; CHECK-NEXT: ret i32 %mod
 %mod = urem i32 %x, %n
 %mod1 = urem i32 %mod, %n
 ret i32 %mod1
}

define i32 @rem3(i32 %x, i32 %n) {
; CHECK-LABEL: @rem3(
; CHECK-NEXT: %[[srem:.*]] = srem i32 %x, %n
; CHECK-NEXT: %[[urem:.*]] = urem i32 %[[srem]], %n
; CHECK-NEXT: ret i32 %[[urem]]
 %mod = srem i32 %x, %n
 %mod1 = urem i32 %mod, %n
 ret i32 %mod1
}
