; RUN: opt %s -instcombine -S | FileCheck %s

declare {i8, i1} @llvm.uadd.with.overflow.i8(i8, i8)

define i8 @test1(i8 %A, i8 %B) {
  %x = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 %A, i8 %B)
  %y = extractvalue  {i8, i1} %x, 0
  ret i8 %y
; CHECK: @test1
; CHECK-NEXT: %y = add i8 %A, %B
; CHECK-NEXT: ret i8 %y
}
