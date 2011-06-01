; Test that vectors are scalarized/lowered correctly.
; RUN: llc -march=x86 -promote-elements < %s | FileCheck %s

; This test is the poster-child for integer-element-promotion.
; Until this feature is complete, we mark this test as expected to fail.
; XFAIL: *
; CHECK: vector_code
; CHECK: ret
define <4 x float> @vector_code(<4 x i64> %A, <4 x i64> %B, <4 x float> %R0, <4 x float> %R1 )  {
   %C = icmp eq <4 x i64> %A, %B
   %K = xor <4 x i1> <i1 1, i1 1, i1 1, i1 1>, %C
   %D = select <4 x i1> %K, <4 x float> %R1, <4 x float> %R0
   ret <4 x float> %D
}

