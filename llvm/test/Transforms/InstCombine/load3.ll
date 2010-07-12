; RUN: opt < %s -instcombine -S | FileCheck %s

; Instcombine should be able to do trivial CSE of loads.

define i32 @test1(i32* %p) {
  %t0 = getelementptr i32* %p, i32 1
  %y = load i32* %t0
  %t1 = getelementptr i32* %p, i32 1
  %x = load i32* %t1
  %a = sub i32 %y, %x
  ret i32 %a
; CHECK: @test1
; CHECK: ret i32 0
}
