; RUN: opt < %s -instcombine -S | FileCheck %s
define i32 @foo(i32 %x, i32 %y) {
; CHECK: @foo
  %add = add nsw i32 %y, %x
  %mul = mul nsw i32 %add, %y
  %square = mul nsw i32 %y, %y
  %res = sub i32 %mul, %square
; CHECK: %res = mul i32 %x, %y
  ret i32 %res
; CHECK: ret i32 %res
}
