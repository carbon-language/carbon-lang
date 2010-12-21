; RUN: opt < %s -instsimplify -S | FileCheck %s

define i32 @add1(i32 %x) {
; CHECK: @add1
; (X + -1) + 1 -> X
  %l = add i32 %x, -1
  %r = add i32 %l, 1
  ret i32 %r
; CHECK: ret i32 %x
}

define i32 @and1(i32 %x, i32 %y) {
; CHECK: @and1
; (X & Y) & X -> X & Y
  %l = and i32 %x, %y
  %r = and i32 %l, %x
  ret i32 %r
; CHECK: ret i32 %l
}

define i32 @and2(i32 %x, i32 %y) {
; CHECK: @and2
; X & (X & Y) -> X & Y
  %r = and i32 %x, %y
  %l = and i32 %x, %r
  ret i32 %l
; CHECK: ret i32 %r
}

define i32 @or1(i32 %x, i32 %y) {
; CHECK: @or1
; (X | Y) | X -> X | Y
  %l = or i32 %x, %y
  %r = or i32 %l, %x
  ret i32 %r
; CHECK: ret i32 %l
}

define i32 @or2(i32 %x, i32 %y) {
; CHECK: @or2
; X | (X | Y) -> X | Y
  %r = or i32 %x, %y
  %l = or i32 %x, %r
  ret i32 %l
; CHECK: ret i32 %r
}

define i32 @xor1(i32 %x, i32 %y) {
; CHECK: @xor1
; (X ^ Y) ^ X = Y
  %l = xor i32 %x, %y
  %r = xor i32 %l, %x
  ret i32 %r
; CHECK: ret i32 %y
}

define i32 @xor2(i32 %x, i32 %y) {
; CHECK: @xor2
; X ^ (X ^ Y) = Y
  %r = xor i32 %x, %y
  %l = xor i32 %x, %r
  ret i32 %l
; CHECK: ret i32 %y
}
