; RUN: llc < %s -march=x86-64 -mcpu=core2 | FileCheck %s

define <4 x i32> @test1(<4 x i32> %x, <4 x i32> %y) {
  %m = mul <4 x i32> %x, %y
  ret <4 x i32> %m
; CHECK-LABEL: test1:
; CHECK: pshufd $49
; CHECK: pmuludq
; CHECK: pshufd $49
; CHECK: pmuludq
; CHECK: shufps $-120
; CHECK: pshufd $-40
; CHECK: ret
}
