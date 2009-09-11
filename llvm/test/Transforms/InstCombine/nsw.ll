; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: define i32 @foo
; %y = sub i32 0, %x
; %z = sdiv i32 %y, 337
; ret i32 %y
define i32 @foo(i32 %x) {
  %y = sub i32 0, %x
  %z = sdiv i32 %y, 337
  ret i32 %y
}

; CHECK: define i32 @bar
; %y = sdiv i32 %x, -337
; ret i32 %y
define i32 @bar(i32 %x) {
  %y = sub nsw i32 0, %x
  %z = sdiv i32 %y, 337
  ret i32 %y
}
