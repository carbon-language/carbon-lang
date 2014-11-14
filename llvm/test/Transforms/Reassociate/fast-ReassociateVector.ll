; RUN: opt < %s -reassociate -S | FileCheck %s

; Canonicalize operands, but don't optimize floating point vector operations.
define <4 x float> @test1() {
; CHECK-LABEL: test1
; CHECK-NEXT: %tmp1 = fsub fast <4 x float> zeroinitializer, zeroinitializer
; CHECK-NEXT: %tmp2 = fmul fast <4 x float> %tmp1, zeroinitializer

  %tmp1 = fsub fast <4 x float> zeroinitializer, zeroinitializer
  %tmp2 = fmul fast <4 x float> zeroinitializer, %tmp1
  ret <4 x float> %tmp2
}

; Commute integer vector operations.
define <2 x i32> @test2(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: test2
; CHECK-NEXT: %tmp1 = add <2 x i32> %x, %y
; CHECK-NEXT: %tmp2 = add <2 x i32> %x, %y
; CHECK-NEXT: %tmp3 = add <2 x i32> %tmp1, %tmp2

  %tmp1 = add <2 x i32> %x, %y
  %tmp2 = add <2 x i32> %y, %x
  %tmp3 = add <2 x i32> %tmp1, %tmp2
  ret <2 x i32> %tmp3
}

define <2 x i32> @test3(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: test3
; CHECK-NEXT: %tmp1 = mul <2 x i32> %x, %y
; CHECK-NEXT: %tmp2 = mul <2 x i32> %x, %y
; CHECK-NEXT: %tmp3 = mul <2 x i32> %tmp1, %tmp2

  %tmp1 = mul <2 x i32> %x, %y
  %tmp2 = mul <2 x i32> %y, %x
  %tmp3 = mul <2 x i32> %tmp1, %tmp2
  ret <2 x i32> %tmp3
}

define <2 x i32> @test4(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: test4
; CHECK-NEXT: %tmp1 = and <2 x i32> %x, %y
; CHECK-NEXT: %tmp2 = and <2 x i32> %x, %y
; CHECK-NEXT: %tmp3 = and <2 x i32> %tmp1, %tmp2

  %tmp1 = and <2 x i32> %x, %y
  %tmp2 = and <2 x i32> %y, %x
  %tmp3 = and <2 x i32> %tmp1, %tmp2
  ret <2 x i32> %tmp3
}

define <2 x i32> @test5(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: test5
; CHECK-NEXT: %tmp1 = or <2 x i32> %x, %y
; CHECK-NEXT: %tmp2 = or <2 x i32> %x, %y
; CHECK-NEXT: %tmp3 = or <2 x i32> %tmp1, %tmp2

  %tmp1 = or <2 x i32> %x, %y
  %tmp2 = or <2 x i32> %y, %x
  %tmp3 = or <2 x i32> %tmp1, %tmp2
  ret <2 x i32> %tmp3
}

define <2 x i32> @test6(<2 x i32> %x, <2 x i32> %y) {
; CHECK-LABEL: test6
; CHECK-NEXT: %tmp1 = xor <2 x i32> %x, %y
; CHECK-NEXT: %tmp2 = xor <2 x i32> %x, %y
; CHECK-NEXT: %tmp3 = xor <2 x i32> %tmp1, %tmp2

  %tmp1 = xor <2 x i32> %x, %y
  %tmp2 = xor <2 x i32> %y, %x
  %tmp3 = xor <2 x i32> %tmp1, %tmp2
  ret <2 x i32> %tmp3
}
