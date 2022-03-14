; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: vaddub

define <8 x i8> @t_i8x8(<8 x i8> %a, <8 x i8> %b) nounwind {
entry:
	%0 = add <8 x i8> %a, %b
	ret <8 x i8> %0
}
