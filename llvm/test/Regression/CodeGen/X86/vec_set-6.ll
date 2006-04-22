; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep unpcklps | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep shufps   | wc -l | grep 1

<4 x float> %test(float %a, float %b, float %c) {
	%tmp = insertelement <4 x float> zeroinitializer, float %a, uint 1
	%tmp8 = insertelement <4 x float> %tmp, float %b, uint 2
	%tmp10 = insertelement <4 x float> %tmp8, float %c, uint 3
	ret <4 x float> %tmp10
}
