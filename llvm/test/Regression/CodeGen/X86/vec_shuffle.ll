; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep shufp | wc -l | grep 1
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movlhps

void %test_v4sf(<4 x float>* %P, float %X, float %Y) {
	%tmp = insertelement <4 x float> zeroinitializer, float %X, uint 0
	%tmp2 = insertelement <4 x float> %tmp, float %X, uint 1
	%tmp4 = insertelement <4 x float> %tmp2, float %Y, uint 2
	%tmp6 = insertelement <4 x float> %tmp4, float %Y, uint 3
	store <4 x float> %tmp6, <4 x float>* %P
	ret void
}

void %test_v2sd(<2 x double>* %P, double %X, double %Y) {
	%tmp = insertelement <2 x double> zeroinitializer, double %X, uint 0
	%tmp2 = insertelement <2 x double> %tmp, double %Y, uint 1
	store <2 x double> %tmp2, <2 x double>* %P
	ret void
}
