; RUN: opt -instcombine -S < %s | FileCheck %s

define i32 @test1(i32 %x) nounwind {
  %and = and i32 %x, 31
  %sub = sub i32 63, %and
  ret i32 %sub

; CHECK: @test1
; CHECK-NEXT: and i32 %x, 31
; CHECK-NEXT: xor i32 %and, 63
; CHECK-NEXT: ret
}
