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

define i32 @select3(i32 %x, i32 %n) {
; CHECK-LABEL: @select3(
; CHECK-NEXT: %mod = srem i32 %x, %n
; CHECK-NEXT: ret i32 %mod
 %mod = srem i32 %x, %n
 %mod1 = srem i32 %mod, %n
 ret i32 %mod1
}
