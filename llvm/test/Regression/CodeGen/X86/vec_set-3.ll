; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep shufps | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pshufd | wc -l | grep 1

<4 x float> %test(float %a) {
	%tmp = insertelement <4 x float> zeroinitializer, float %a, uint 1
	%tmp5 = insertelement <4 x float> %tmp, float 0.000000e+00, uint 2
	%tmp6 = insertelement <4 x float> %tmp5, float 0.000000e+00, uint 3
	ret <4 x float> %tmp6
}

<2 x long> %test(int %a) {
	%tmp7 = insertelement <4 x int> zeroinitializer, int %a, uint 2
	%tmp9 = insertelement <4 x int> %tmp7, int 0, uint 3
	%tmp10 = cast <4 x int> %tmp9 to <2 x long>
	ret <2 x long> %tmp10
}
