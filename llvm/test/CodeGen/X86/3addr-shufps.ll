; RUN: llc < %s -mtriple=x86_64-apple-darwin13 -mcpu=pentium4 | FileCheck %s

define <4 x float> @test1(<4 x i32>, <4 x float> %b) {
  %s = shufflevector <4 x float> %b, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 2, i32 3>
  ret <4 x float> %s

; We convert shufps -> pshufd here to save a move.
; CHECK-LABEL: test1:
; CHECK:         pshufd $-27, %xmm1, %xmm0
; CHECK-NEXT:    ret
}
