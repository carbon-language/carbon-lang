; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movlhps   | wc -l | grep 2 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep unpcklps  | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep punpckldq | wc -l | grep 1

<4 x float> %test1(float %a, float %b) {
	%tmp = insertelement <4 x float> zeroinitializer, float %a, uint 0
	%tmp6 = insertelement <4 x float> %tmp, float 0.000000e+00, uint 1
	%tmp8 = insertelement <4 x float> %tmp6, float %b, uint 2
	%tmp9 = insertelement <4 x float> %tmp8, float 0.000000e+00, uint 3
	ret <4 x float> %tmp9
}

<4 x float> %test2(float %a, float %b) {
	%tmp = insertelement <4 x float> zeroinitializer, float %a, uint 0
	%tmp7 = insertelement <4 x float> %tmp, float %b, uint 1
	%tmp8 = insertelement <4 x float> %tmp7, float 0.000000e+00, uint 2
	%tmp9 = insertelement <4 x float> %tmp8, float 0.000000e+00, uint 3
	ret <4 x float> %tmp9
}

<2 x long> %test3(int %a, int %b) {
	%tmp = insertelement <4 x int> zeroinitializer, int %a, uint 0
	%tmp6 = insertelement <4 x int> %tmp, int %b, uint 1
	%tmp8 = insertelement <4 x int> %tmp6, int 0, uint 2
	%tmp10 = insertelement <4 x int> %tmp8, int 0, uint 3
	%tmp11 = cast <4 x int> %tmp10 to <2 x long>
	ret <2 x long> %tmp11
}
