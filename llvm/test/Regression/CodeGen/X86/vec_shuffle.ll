; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep shufp | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movups | wc -l | grep 1 &&
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep pshufhw | wc -l | grep 1

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

void %test_v8i16(<2 x long>* %res, <2 x long>* %A) {
	%tmp = load <2 x long>* %A
	%tmp = cast <2 x long> %tmp to <8 x short>
	%tmp = extractelement <8 x short> %tmp, uint 0
	%tmp1 = extractelement <8 x short> %tmp, uint 1
	%tmp2 = extractelement <8 x short> %tmp, uint 2
	%tmp3 = extractelement <8 x short> %tmp, uint 3
	%tmp4 = extractelement <8 x short> %tmp, uint 6
	%tmp5 = extractelement <8 x short> %tmp, uint 5
	%tmp6 = extractelement <8 x short> %tmp, uint 4
	%tmp7 = extractelement <8 x short> %tmp, uint 7
	%tmp8 = insertelement <8 x short> undef, short %tmp, uint 0
	%tmp9 = insertelement <8 x short> %tmp8, short %tmp1, uint 1
	%tmp10 = insertelement <8 x short> %tmp9, short %tmp2, uint 2
	%tmp11 = insertelement <8 x short> %tmp10, short %tmp3, uint 3
	%tmp12 = insertelement <8 x short> %tmp11, short %tmp4, uint 4
	%tmp13 = insertelement <8 x short> %tmp12, short %tmp5, uint 5
	%tmp14 = insertelement <8 x short> %tmp13, short %tmp6, uint 6
	%tmp15 = insertelement <8 x short> %tmp14, short %tmp7, uint 7
	%tmp15 = cast <8 x short> %tmp15 to <2 x long>
	store <2 x long> %tmp15, <2 x long>* %res
	ret void
}
