; There should be exactly one shift and one add left.
; RUN: opt < %s -reassociate -instcombine -S | FileCheck %s

define i32 @test1(i32 %X, i32 %Y) {
; CHECK-LABEL: test1
; CHECK-NEXT: %tmp = add i32 %Y, %X
; CHECK-NEXT: %tmp1 = shl i32 %tmp, 1
; CHECK-NEXT: ret i32 %tmp1

  %tmp.2 = shl i32 %X, 1
  %tmp.6 = shl i32 %Y, 1
  %tmp.4 = add i32 %tmp.6, %tmp.2
  ret i32 %tmp.4
}
