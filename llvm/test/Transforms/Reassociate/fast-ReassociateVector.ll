; RUN: opt < %s -reassociate -S | FileCheck %s

; Don't handle floating point vector operations.
define <4 x float> @test1() {
; CHECK-LABEL: test1
; CHECK-NEXT: %tmp1 = fsub fast <4 x float> zeroinitializer, zeroinitializer
; CHECK-NEXT: %tmp2 = fmul fast <4 x float> zeroinitializer, %tmp1

  %tmp1 = fsub fast <4 x float> zeroinitializer, zeroinitializer
  %tmp2 = fmul fast <4 x float> zeroinitializer, %tmp1
  ret <4 x float> %tmp2
}

; We don't currently commute integer vector operations.
define <2 x i32> @test2(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: test2
; CHECK-NEXT: %tmp1 = add <2 x i32> %x, %y
; CHECK-NEXT: %tmp2 = add <2 x i32> %y, %x
; CHECK-NEXT: %tmp3 = add <2 x i32> %tmp1, %tmp2

  %tmp1 = add <2 x i32> %x, %y
  %tmp2 = add <2 x i32> %y, %x
  %tmp3 = add <2 x i32> %tmp1, %tmp2
  ret <2 x i32> %tmp3
}
