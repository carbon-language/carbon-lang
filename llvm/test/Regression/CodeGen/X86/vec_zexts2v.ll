; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movss | wc -l | grep 2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep xorps | wc -l | grep 1

void %test1(<4 x float>* %b, float %X) {
	%tmp = insertelement <4 x float> zeroinitializer, float %X, uint 0
	%tmp1 = insertelement <4 x float> %tmp, float 0.000000e+00, uint 1
	%tmp2 = insertelement <4 x float> %tmp1, float 0.000000e+00, uint 2
	%tmp3 = insertelement <4 x float> %tmp2, float 0.000000e+00, uint 3
	store <4 x float> %tmp3, <4 x float>* %b
	ret void
}

void %test2(<4 x float>* %b, float %X, float %Y) {
	%tmp2 = mul float %X, %Y
	%tmp = insertelement <4 x float> zeroinitializer, float %tmp2, uint 0
	%tmp4 = insertelement <4 x float> %tmp, float 0.000000e+00, uint 1
	%tmp5 = insertelement <4 x float> %tmp4, float 0.000000e+00, uint 2
	%tmp6 = insertelement <4 x float> %tmp5, float 0.000000e+00, uint 3
	store <4 x float> %tmp6, <4 x float>* %b
	ret void
}
