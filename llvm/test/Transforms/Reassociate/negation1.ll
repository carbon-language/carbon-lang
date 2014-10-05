; RUN: opt < %s -reassociate -instcombine -S | FileCheck %s

; Test that we can turn things like A*B + X - A*B -> X.

define i32 @test1(i32 %a, i32 %b, i32 %x) {
; CHECK-LABEL: test1
; CHECK: ret i32 %x

  %c = mul i32 %a, %b
  %d = add i32 %c, %x
  %c1 = mul i32 %a, %b
  %f = sub i32 %d, %c1
  ret i32 %f
}

