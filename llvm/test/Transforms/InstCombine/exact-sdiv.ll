; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: define i32 @foo
; CHECK: sdiv i32 %x, 8
define i32 @foo(i32 %x) {
  %y = sdiv i32 %x, 8
  ret i32 %y
}

; CHECK: define i32 @bar
; CHECK: ashr i32 %x, 3
define i32 @bar(i32 %x) {
  %y = sdiv exact i32 %x, 8
  ret i32 %y
}

; CHECK: i32 @a0
; CHECK: %y = srem i32 %x, 3
; CHECK: %z = sub i32 %x, %y
; CHECK: ret i32 %z
define i32 @a0(i32 %x) {
  %y = sdiv i32 %x, 3
  %z = mul i32 %y, 3
  ret i32 %z
}

; CHECK: i32 @b0
; CHECK: ret i32 %x
define i32 @b0(i32 %x) {
  %y = sdiv exact i32 %x, 3
  %z = mul i32 %y, 3
  ret i32 %z
}

; CHECK: i32 @a1
; CHECK: %y = srem i32 %x, 3
; CHECK: %z = sub i32 %y, %x
; CHECK: ret i32 %z
define i32 @a1(i32 %x) {
  %y = sdiv i32 %x, 3
  %z = mul i32 %y, -3
  ret i32 %z
}

; CHECK: i32 @b1
; CHECK: %z = sub i32 0, %x
; CHECK: ret i32 %z
define i32 @b1(i32 %x) {
  %y = sdiv exact i32 %x, 3
  %z = mul i32 %y, -3
  ret i32 %z
}
