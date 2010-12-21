; RUN: opt < %s -instsimplify -S | FileCheck %s

define i32 @factorize(i32 %x, i32 %y) {
; CHECK: @factorize
; (X | 1) & (X | 2) -> X | (1 & 2) -> X
  %l = or i32 %x, 1
  %r = or i32 %x, 2
  %z = and i32 %l, %r
  ret i32 %z
; CHECK: ret i32 %x
}

define i32 @factorize2(i32 %x) {
; CHECK: @factorize2
; 3*X - 2*X -> X
  %l = mul i32 3, %x
  %r = mul i32 2, %x
  %z = sub i32 %l, %r
  ret i32 %z
; CHECK: ret i32 %x
}

define i32 @expand(i32 %x) {
; CHECK: @expand
; ((X & 1) | 2) & 1 -> ((X & 1) & 1) | (2 & 1) -> (X & 1) | 0 -> X & 1
  %a = and i32 %x, 1
  %b = or i32 %a, 2
  %c = and i32 %b, 1
  ret i32 %c
; CHECK: ret i32 %a
}
