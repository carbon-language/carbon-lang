; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep shufps &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movddup

void %test_v4sf(<4 x float>* %P, <4 x float>* %Q, float %X) {
	%tmp = insertelement <4 x float> zeroinitializer, float %X, uint 0
	%tmp2 = insertelement <4 x float> %tmp, float %X, uint 1
	%tmp4 = insertelement <4 x float> %tmp2, float %X, uint 2
	%tmp6 = insertelement <4 x float> %tmp4, float %X, uint 3
	%tmp8 = load <4 x float>* %Q
	%tmp10 = mul <4 x float> %tmp8, %tmp6
	store <4 x float> %tmp10, <4 x float>* %P
	ret void
}

void %test_v2sd(<2 x double>* %P, <2 x double>* %Q, double %X) {
	%tmp = insertelement <2 x double> zeroinitializer, double %X, uint 0
	%tmp2 = insertelement <2 x double> %tmp, double %X, uint 1
	%tmp4 = load <2 x double>* %Q
	%tmp6 = mul <2 x double> %tmp4, %tmp2
	store <2 x double> %tmp6, <2 x double>* %P
	ret void
}
