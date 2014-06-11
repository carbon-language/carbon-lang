; RUN: opt < %s -reassociate -instcombine -S | FileCheck %s

; Test that we can turn things like X*-(Y*Z) -> X*-1*Y*Z.

define i32 @test1(i32 %a, i32 %b, i32 %z) {
; CHECK-LABEL: test1
; CHECK-NEXT: %e = mul i32 %a, 12345
; CHECK-NEXT: %f = mul i32 %e, %b
; CHECK-NEXT: %g = mul i32 %f, %z
; CHECK-NEXT: ret i32 %g

  %c = sub i32 0, %z
  %d = mul i32 %a, %b
  %e = mul i32 %c, %d
  %f = mul i32 %e, 12345
  %g = sub i32 0, %f
  ret i32 %g
}

define i32 @test2(i32 %a, i32 %b, i32 %z) {
; CHECK-LABEL: test2
; CHECK-NEXT: %e = mul i32 %a, 40
; CHECK-NEXT: %f = mul i32 %e, %z
; CHECK-NEXT: ret i32 %f

  %d = mul i32 %z, 40
  %c = sub i32 0, %d
  %e = mul i32 %a, %c
  %f = sub i32 0, %e
  ret i32 %f
}
