; RUN: llc < %s -march=x86 | FileCheck %s

; When legalizing the v4i1 constant, we need to consider the boolean contents
; For x86 a boolean vector constant is all ones so the constants in memory
; will be ~0U not 1.

; CHECK: .long	4294967295
; CHECK: .long	4294967295
; CHECK: .long	0
; CHECK: .long	0

; CHECK: test
define <4 x i8> @test(<4 x i8> %a, <4 x i8> %b) {
  %sel = select <4 x i1> <i1 true, i1 true, i1 false, i1 false>, <4 x i8> %a, <4 x i8> %b
	ret <4 x i8> %sel
}
